import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import json
from rapidfuzz import process, fuzz
import os
import warnings
from datetime import date

# ignorando warnings não cruciais para a saída limpa
warnings.filterwarnings('ignore')

# mapeando identificadores de ligas para as confederações correspondentes
_UEFA_LEAGUES     = {'ES1','GB1','IT1','FR1','L1','NL1','BE1','PT1','TR1','RU1','GR1','SC1','DK1','SE1'}
_CONMEBOL_LEAGUES = {'BRA1','ARG1','COL1','CHI1','ECU1','URU1','VEN1','PER1','PAR1','BOL1'}
_CONCACAF_LEAGUES = {'MLS1','MEX1','JAP1','KOR1'}

# padronizando as ligas para reduzir a dimensão das features de confederação
def _confederation(domestic_competition_id: str) -> str:
    lid = str(domestic_competition_id).upper().strip()
    if lid in _UEFA_LEAGUES:     return 'UEFA'
    if lid in _CONMEBOL_LEAGUES: return 'CONMEBOL'
    if lid in _CONCACAF_LEAGUES: return 'CONCACAF'
    return lid

# abstraindo todo o pipeline de inferência num motor unificado
class TransferPredictor:
    """
    Motor de inferência para prever valores de mercado e custos de transferência.

    Casos de uso:
    - predict_current_value(player_id)               → Valor de mercado atual do jogador
    - predict_transfer_fee(player_id, buyer, seller, date) → Faixa de custo de transferência
    """

    def __init__(self, project_dir="."):
        # mapeando os caminhos relativos ao diretório raiz de execução
        self.data_dir   = os.path.join(project_dir, "processing", "data")
        self.models_dir = os.path.join(project_dir, "processing", "models")

        # preparando as referências vazias dos preditores xgboost
        self.model1     = None
        self.model2_p15 = None
        self.model2_p50 = None
        self.model2_p85 = None

        # iniciando dicionários que ditarão os schemas e multiplicadores nos cálculos
        self.features1   = None
        self.features2   = None
        self.club_ratios = None

        # definindo o baseline do score usado no ajuste matemático posterior
        self.base_score_m1 = 0.0

        # instanciando espaços vazios pros frames parquets processados
        self.players_df     = None
        self.clubs_df       = None
        self.appearances_df = None

        # declarando referências temporais para as avaliações de clubes
        self.player_club_value_ts     = None
        self.player_club_value_latest = None

        # guardando o estado do loader para prevenir overhead de re-leitura
        self._is_loaded = False

    def _get_base_score(self, path: str) -> float:
        """Lê o base_score do JSON do modelo XGBoost."""
        with open(path, 'r') as f:
            data = json.load(f)
        val = data['learner']['learner_model_param']['base_score']
        return float(val.strip('[]'))

    def load_data(self):
        """Carrega modelos e datasets na memória (Lazy Loading)."""
        if self._is_loaded:
            return

        # injetando na memória dicionários de features e transformadores treinados
        with open(os.path.join(self.models_dir, "feature_names_model1.pkl"), "rb") as f:
            self.features1 = pickle.load(f)
        with open(os.path.join(self.models_dir, "feature_names_model2.pkl"), "rb") as f:
            self.features2 = pickle.load(f)
        with open(os.path.join(self.models_dir, "club_ratios.pkl"), "rb") as f:
            self.club_ratios = pickle.load(f)

        # subindo a arquitetura do modelo de avaliação intrínseca (modelo 1)
        path_m1 = os.path.join(self.models_dir, "model1_market_value.json")
        self.model1 = xgb.Booster()
        self.model1.load_model(path_m1)
        self.base_score_m1 = self._get_base_score(path_m1)

        # subindo as ramificações p15 p50 p85 relativas a simulação de fee (modelo 2)
        path_p15 = os.path.join(self.models_dir, "model2_transfer_fee_p15.json")
        self.model2_p15 = xgb.Booster()
        self.model2_p15.load_model(path_p15)

        path_p50 = os.path.join(self.models_dir, "model2_transfer_fee_p50.json")
        self.model2_p50 = xgb.Booster()
        self.model2_p50.load_model(path_p50)

        path_p85 = os.path.join(self.models_dir, "model2_transfer_fee_p85.json")
        self.model2_p85 = xgb.Booster()
        self.model2_p85.load_model(path_p85)

        # consumindo a base processada de características e biometria dos atletas
        self.players_df = pd.read_parquet(os.path.join(self.data_dir, "players_processed.parquet"))
        self.players_df = self.players_df.set_index('player_id', drop=False)

        # consumindo e indexando catálogo de clubes global
        self.clubs_df = pd.read_parquet(os.path.join(self.data_dir, "clubs_processed.parquet"))
        self.clubs_df = self.clubs_df.set_index('club_id', drop=False)

        # engatando o dataframe com todas as aparições sazonais para métricas relativas
        self.appearances_df = pd.read_parquet(os.path.join(self.data_dir, "appearances_by_season.parquet"))

        # compondo histórico de status financeiro dos clubes envolvidos por ano
        m1_df = pd.read_parquet(
            os.path.join(self.data_dir, "model1_dataset.parquet"),
            columns=['player_id', 'year', 'club_computed_market_value']
        )
        self.player_club_value_ts = (
            m1_df.groupby(['player_id', 'year'])['club_computed_market_value'].last()
        )
        self.player_club_value_latest = (
            m1_df.groupby('player_id')['club_computed_market_value'].last().to_dict()
        )

        # selando o estado como carregado e acessível
        self._is_loaded = True

    def _get_player_row(self, player_id: int) -> pd.Series:
        # validando existência do jogador para não quebrar a chain
        if player_id not in self.players_df.index:
            raise ValueError(f"Jogador com ID {player_id} não encontrado.")
        row = self.players_df.loc[player_id]
        return row.iloc[0] if isinstance(row, pd.DataFrame) else row

    def _get_club_row(self, club_id) -> pd.Series:
        # devolvendo nans padronizados para o shape bater em caso de clube inexistente
        if pd.isna(club_id) or club_id not in self.clubs_df.index:
            return pd.Series(index=self.clubs_df.columns, dtype=float)
        row = self.clubs_df.loc[club_id]
        return row.iloc[0] if isinstance(row, pd.DataFrame) else row

    def _get_club_computed_value(self, player_id: int, year: int) -> float:
        """
        Retorna o club_computed_market_value para o ano mais próximo disponível,
        sem vazar dados futuros (sempre pega o mais recente <= year pedido).
        """
        pid = int(player_id)
        
        # procurando o valor direto para a chave composta
        try:
            ts = self.player_club_value_ts
            if (pid, year) in ts.index:
                return float(ts.loc[(pid, year)])
            
            # fallback para encontrar o ano antecedente mais próximo existente
            if pid in ts.index.get_level_values(0):
                past = [y for y in ts.loc[pid].index if y <= year]
                if past:
                    return float(ts.loc[(pid, max(past))])
        except (KeyError, TypeError):
            pass
        
        # resolvendo valores default caso falhe tudo (valor genérico fixado)
        return float(self.player_club_value_latest.get(pid, 10_000_000))

    def _get_app_stats(self, player_apps: pd.DataFrame, season_year: int) -> dict:
        """Retorna stats de aparições de uma temporada. Zeros se sem dados."""
        rows = player_apps[player_apps['season_year'] == season_year]
        
        # forçando os totais nulos caso o filtro retorne vazio
        if rows.empty:
            return {
                'goals_season': 0, 'assists_season': 0, 'minutes_season': 0,
                'games_season': 0, 'goals_per_90': 0, 'assists_per_90': 0,
                'minutes_per_game': 0, 'yellow_cards_season': 0, 'red_cards_season': 0
            }
            
        # empacotando apenas os índices requisitados pro dicionário de estado
        r = rows.iloc[0]
        return {k: r.get(k, 0) for k in [
            'goals_season', 'assists_season', 'minutes_season', 'games_season',
            'goals_per_90', 'assists_per_90', 'minutes_per_game',
            'yellow_cards_season', 'red_cards_season'
        ]}

    def _build_m1_features(self, player_row, club_row, app_stats, year, dob) -> pd.DataFrame:
        """Monta o vetor de features para o Modelo 1."""
        age = year - dob.year
        
        # acumulando todos os traços numa visão unificada para input
        feature_dict = player_row.to_dict()
        feature_dict.update(club_row.to_dict())
        feature_dict.update(app_stats)
        
        # injetando as composições derivadas de tempo e idade
        feature_dict['age']         = age
        feature_dict['age_squared'] = age ** 2
        feature_dict['year']        = year
        feature_dict['club_computed_market_value'] = self._get_club_computed_value(
            int(player_row['player_id']), year
        )
        
        # transformando pro schema de array e fixando as colunas faltantes em 0
        X = pd.DataFrame([feature_dict])
        for col in self.features1:
            if col not in X.columns:
                X[col] = 0
                
        # retornando o recorte estrito de colunas que o xgboost exige
        return X[self.features1].astype(float)

    def _predict_m1(self, X: pd.DataFrame) -> float:
        # mapeando os valores em estrutura dmatrix nativa da engine
        dmatrix  = xgb.DMatrix(X)
        pred_log = self.model1.predict(dmatrix)[0]
        
        # desfazendo a escala logarítmica exposta pelo modelo durante sua regressão
        return float(np.expm1(pred_log))

    def search_player(self, query: str, limit: int = 5) -> list:
        """Busca fuzzy por nome de jogador."""
        self.load_data()
        names_dict = self.players_df['name'].dropna().to_dict()
        
        # rodando a inferência na engine rapidfuzz pro pareamento descritivo
        results = process.extract(query, names_dict, scorer=fuzz.WRatio, limit=limit)
        
        # iterando e mesclando o clube atual com o resultado isolado
        matches = []
        for name, score, idx in results:
            row      = self.players_df.loc[idx]
            club_id  = row.get('current_club_id')
            club_name = "Sem Clube"
            if pd.notna(club_id) and club_id in self.clubs_df.index:
                club_name = self.clubs_df.loc[club_id]['name']
            
            # concatenando os match dictionaries pro array público
            matches.append({
                "player_id":    int(row['player_id']),
                "name":         name,
                "current_club": club_name,
                "score":        score
            })
        return matches

    def search_club(self, query: str, limit: int = 5) -> list:
        """Busca fuzzy por nome de clube."""
        self.load_data()
        names_dict = self.clubs_df['name'].dropna().to_dict()
        
        # chamando a rotina principal para resolver os limites do texto
        results = process.extract(query, names_dict, scorer=fuzz.WRatio, limit=limit)
        
        # destilando os metadados mais cruciais para ajudar na exibição do terminal
        matches = []
        for name, score, idx in results:
            row = self.clubs_df.loc[idx]
            matches.append({
                "club_id": int(row['club_id']),
                "name":    name,
                "league":  row.get('domestic_competition_id', 'Unknown'),
                "score":   score
            })
        return matches

    def predict_current_value(self, player_id: int) -> float:
        """
        Prevê o valor de mercado ATUAL do jogador em euros.

        Usa os stats da temporada mais recente disponível no dataset
        e avalia no contexto econômico do ano corrente.
        """
        self.load_data()

        # recuperando estaticamente as biografias cruciais
        player_row   = self._get_player_row(player_id)
        club_row     = self._get_club_row(player_row.get('current_club_id'))
        dob          = pd.to_datetime(player_row['date_of_birth'])
        current_year = date.today().year

        # achando a última janela de aparição reportada pra evitar drifts nulos
        player_apps   = self.appearances_df[self.appearances_df['player_id'] == player_id]
        latest_season = int(player_apps['season_year'].max()) if not player_apps.empty else current_year - 1

        # construindo matriz e delegando pra função interna disparar
        app_stats = self._get_app_stats(player_apps, latest_season)
        X = self._build_m1_features(player_row, club_row, app_stats, current_year, dob)
        return self._predict_m1(X)

    def predict_transfer_fee(
        self,
        player_id:    int,
        buyer_club_id:  int,
        seller_club_id: int,
        transfer_date:  str
    ) -> dict:
        """
        Estima a faixa de custo de transferência [P15, P50, P85] para uma negociação.

        Parâmetros:
            player_id:      ID do jogador
            buyer_club_id:  ID do clube comprador
            seller_club_id: ID do clube vendedor
            transfer_date:  Data da transferência (YYYY-MM-DD); usa hoje se vazio

        Retorna dict com:
            market_value_eur:     valor de mercado atual usado como base
            transfer_fee_p15_eur: cenário pessimista (15% das negociações ficam abaixo)
            transfer_fee_p50_eur: cenário esperado (mediana)
            transfer_fee_p85_eur: cenário otimista (15% ficam acima)
        """
        self.load_data()

        # gerando a propensão de valor com o m1 como âncora matemática de features
        pred_mv = self.predict_current_value(player_id)

        # abstraindo características biográficas relativas aos envolvidos
        player_row = self._get_player_row(player_id)
        buyer_row  = self._get_club_row(buyer_club_id)
        seller_row = self._get_club_row(seller_club_id)
        dob        = pd.to_datetime(player_row['date_of_birth'])

        # mapeando transfer date e se a movimentação ocorre ou não no verão europeu
        date_obj      = pd.to_datetime(transfer_date)
        transfer_year = date_obj.year
        transfer_window = 1 if date_obj.month in (1, 2, 7, 8) else 0

        # apontando sazonalidade correta antes ou depois da largada de temporada normal
        target_season = transfer_year if date_obj.month >= 8 else transfer_year - 1
        player_apps   = self.appearances_df[self.appearances_df['player_id'] == player_id]

        # resolvendo se a temporada desejada tem os dados ou se é falha
        if not player_apps.empty:
            available   = player_apps['season_year'].unique()
            season_year = target_season if target_season in available else int(player_apps['season_year'].max())
        else:
            season_year = target_season

        # encontrando o marco de maturidade físico durante as transferências
        player_age_at_transfer = transfer_year - dob.year

        # destrinchando proximidades culturais de ligas de confederações
        buyer_conf  = _confederation(buyer_row.get('domestic_competition_id', ''))
        seller_conf = _confederation(seller_row.get('domestic_competition_id', ''))
        same_confederation = 1 if buyer_conf == seller_conf else 0

        # capturando os multiplicadores de negociação de comprador/vendedor globais
        global_buyer_ratio  = self.club_ratios['buyer'].get('global', 1.0)
        global_seller_ratio = self.club_ratios['seller'].get('global', 1.0)
        buyer_ratio  = self.club_ratios['buyer'].get(buyer_club_id, global_buyer_ratio)
        seller_ratio = self.club_ratios['seller'].get(seller_club_id, global_seller_ratio)

        # consumindo predições ofensivas baseadas em aparições relativas à época
        app_rows       = player_apps[player_apps['season_year'] == season_year]
        goals_per_90   = app_rows.iloc[0].get('goals_per_90', 0)   if not app_rows.empty else 0
        assists_per_90 = app_rows.iloc[0].get('assists_per_90', 0) if not app_rows.empty else 0

        # populando o dicionário consolidado final pra injeção no frame
        feature_dict = {
            'predicted_value_m1':          pred_mv,
            'buyer_ratio':                 buyer_ratio,
            'seller_ratio':                seller_ratio,
            'buyer_league_tier':           buyer_row.get('league_tier', 0),
            'seller_league_tier':          seller_row.get('league_tier', 0),
            'same_confederation':          same_confederation,
            'transfer_window':             transfer_window,
            'transfer_year':               transfer_year,
            'player_age_at_transfer':      player_age_at_transfer,
            'position_rank':               player_row.get('position_rank', 0),
            'national_team_ranking_inv':   player_row.get('national_team_ranking_inv', 0),
            'goals_per_90':                goals_per_90,
            'assists_per_90':              assists_per_90,
            'buyer_net_transfer_record':   buyer_row.get('net_transfer_record', 0),
            'buyer_national_team_players': buyer_row.get('national_team_players', 0),
            'buyer_stadium_seats':         buyer_row.get('stadium_seats', 0),
            'seller_net_transfer_record':  seller_row.get('net_transfer_record', 0),
            'seller_national_team_players':seller_row.get('national_team_players', 0),
            'seller_stadium_seats':        seller_row.get('stadium_seats', 0),
        }

        # formatando dataframe e eliminando inconsistências na tipagem
        X = pd.DataFrame([feature_dict])
        for col in self.features2:
            if col not in X.columns:
                X[col] = 0
        X = X[self.features2].astype(float)
        dmatrix = xgb.DMatrix(X)

        # aplicando inferência quântica nas três fronteiras baseada na matriz de features
        p15_log = self.model2_p15.predict(dmatrix)[0]
        p50_log = self.model2_p50.predict(dmatrix)[0]
        p85_log = self.model2_p85.predict(dmatrix)[0]

        # empacotando transformações des-logarítmicas de cada faixa para exposição
        return {
            "market_value_eur":      pred_mv,
            "transfer_fee_p15_eur":  float(np.expm1(p15_log)),
            "transfer_fee_p50_eur":  float(np.expm1(p50_log)),
            "transfer_fee_p85_eur":  float(np.expm1(p85_log)),
        }


# inicialização básica executada caso invocado diretamente pelo python path
if __name__ == "__main__":
    predictor = TransferPredictor()
    res = predictor.search_player("Vinicius Junior", limit=1)
    print("Player Search:", res)
    if res:
        val = predictor.predict_current_value(res[0]['player_id'])
        print(f"Valor Atual: € {val:,.2f}")
