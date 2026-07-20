# market-value-forecasting

## Descrição do Projeto

O market-value-forecasting é uma pipeline de machine learning focada em prever valores de mercado de jogadores de futebol e simular custos de transferências reais. O objetivo é utilizar dados sazonais e de contexto financeiro dos clubes para criar estimativas de mercado precisas.

É possível gerar essas predições de duas formas diferentes. O modelo primário serve para prever o valor intrínseco do jogador, enquanto um segundo modelo simula o valor final pago na transferência, dependendo do clube que compra e do clube que vende. A ideia é utilizar a regressão quantílica no modelo de transferência para prever cenários otimistas e pessimistas, invés de prever apenas um número fixo.

## Instalação e Execução

Para começar, clone o repositório para sua máquina local e acesse o diretório do projeto:

```bash
git clone https://github.com/Liperasz/market-value-forecasting.git
cd market-value-forecasting
```

### Uso completo (Treinamento e Avaliação)

Para rodar todo o pipeline de dados e treinamento dos notebooks, é necessário ter um ambiente Python configurado. 

1. Instale as bibliotecas requeridas:
```bash
pip install -r requirements.txt
```

Isso vai instalar as dependências principais, além de ferramentas como o Jupyter, SHAP e Optuna, que servem para treinamento e avaliação.

### Uso apenas da inferência (CLI)

Um ponto importante é que o projeto possui um terminal interativo simples para quem deseja testar os modelos sem ter que rodar os notebooks inteiros de treinamento. Diferente do treinamento, a inferência precisa de pacotes muito mais leves.

Se o objetivo for rodar apenas a ferramenta de linha de comando (`cli.py`), basta instalar as bibliotecas principais do modelo:

```bash
pip install pandas numpy xgboost rapidfuzz rich
```

Após isso, é preciso apenas executar o comando abaixo:

```bash
python cli.py
```

Isso vai abrir um menu onde é possível pesquisar o nome de jogadores através da busca por similaridade do RapidFuzz, além de simular os custos das transferências. O sistema utiliza um carregamento tardio de variáveis, ou seja, ele só joga os grandes arquivos de dados na memória quando a primeira pesquisa é feita de fato.

## Arquitetura

O sistema funciona através da classe principal chamada TransferPredictor no arquivo predict.py. Através dessa classe é possível instanciar os modelos do XGBoost em conjunto. De forma geral, a ausência de dados sazonais ou de clubes desconhecidos não quebram o código, pois o motor preenche valores nulos de forma automatizada para manter a robustez.
