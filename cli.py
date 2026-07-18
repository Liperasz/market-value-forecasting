import os
import sys
from datetime import date
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, IntPrompt
from rich.rule import Rule
from rich import box
from predict import TransferPredictor

# inicializando o console do rich para renderização estilizada no terminal
console = Console()

# definindo o banner visual que é exibido no topo da aplicação
HEADER = Panel.fit(
    "[bold green]⚽ Football Transfer Intelligence[/bold green]\n"
    "[italic]Motor de Inferência e Avaliação de Mercado[/italic]",
    border_style="green",
    box=box.DOUBLE
)

# limpando a saída do terminal de forma compatível com windows e unix
def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

# imprimindo o cabeçalho no console do usuário
def print_header():
    console.print(HEADER)

# pedindo uma data ao usuário e usando a atual como fallback
def ask_date(prompt_text: str) -> str:
    """Pede uma data ao usuário. Enter sem digitar nada retorna a data de hoje."""
    today = date.today().strftime("%Y-%m-%d")
    raw = Prompt.ask(f"{prompt_text} [dim](YYYY-MM-DD, Enter para hoje)[/dim]", default="")
    return raw.strip() if raw.strip() else today

# gerenciando o fluxo principal da cli e loop de interações
def main():
    clear()
    print_header()

    # carregando instâncias dos modelos e dados na memória
    with console.status("[bold blue]Carregando modelos e dados...", spinner="dots"):
        predictor = TransferPredictor()
        predictor.load_data()

    # mantendo o menu em loop contínuo até ação explícita de saída
    while True:
        console.print()
        console.print(Rule(style="dim"))
        console.print("[bold yellow]Menu Principal[/bold yellow]")
        console.print("  1. 💶  Valor de mercado atual de um jogador")
        console.print("  2. 🤝  Simular custo de transferência entre clubes")
        console.print("  3. ❌  Sair")
        console.print()

        choice = Prompt.ask("Escolha uma opção", choices=["1", "2", "3"])

        # fluxo focado apenas no valor intrínseco de mercado
        if choice == "1":
            # disparando a busca interativa de jogador e interrompendo se cancelado
            player = search_and_select_player(predictor)
            if not player:
                continue

            # realizando a inferência no modelo para determinar valor financeiro
            with console.status("[bold blue]Calculando valor de mercado...", spinner="dots"):
                try:
                    value = predictor.predict_current_value(player['player_id'])
                except Exception as e:
                    console.print(f"[bold red]Erro:[/bold red] {e}")
                    continue

            # atualizando a interface com o resultado da predição
            clear()
            print_header()
            show_current_value_card(player, value)

        # fluxo que requer dois clubes para simular um valor real de transferência
        elif choice == "2":
            # buscando o alvo da negociação
            player = search_and_select_player(predictor)
            if not player:
                continue

            # identificando quem detém os direitos do jogador no cenário
            console.print(f"\n[bold cyan]Clube Vendedor[/bold cyan]")
            seller = search_and_select_club(predictor)
            if not seller:
                continue

            # identificando o clube interessado na compra
            console.print(f"\n[bold cyan]Clube Comprador[/bold cyan]")
            buyer = search_and_select_club(predictor)
            if not buyer:
                continue

            # coletando a data da transferência para referenciar o contexto sazonal correto
            transfer_date = ask_date("Data da Transferência")

            # executando os modelos e estimando as três faixas da negociação
            with console.status("[bold blue]Simulando negociação...", spinner="dots"):
                try:
                    res = predictor.predict_transfer_fee(
                        player['player_id'],
                        buyer['club_id'],
                        seller['club_id'],
                        transfer_date
                    )
                except Exception as e:
                    console.print(f"[bold red]Erro:[/bold red] {e}")
                    continue

            # exibindo o resumo completo da operação calculada
            clear()
            print_header()
            show_transfer_card(player, seller, buyer, transfer_date, res)

        # processando o encerramento voluntário da aplicação
        elif choice == "3":
            clear()
            console.print("[bold green]Até logo![/bold green]")
            break

# abstraindo a lógica de fuzzy search em jogadores com seleção pelo terminal
def search_and_select_player(predictor) -> dict | None:
    query   = Prompt.ask("\n🔍 Jogador (busca fuzzy)")
    results = predictor.search_player(query, limit=5)

    # validando se houve pelo menos uma correspondência na busca
    if not results:
        console.print("[red]Nenhum jogador encontrado.[/red]")
        return None

    # desenhando a tabela com as opções encontradas no índice
    table = Table(box=box.SIMPLE, show_header=True)
    table.add_column("#",     justify="right", style="cyan",    no_wrap=True)
    table.add_column("Nome",                   style="magenta")
    table.add_column("Clube Atual",            style="green")
    table.add_column("Score", justify="right", style="dim")
    for i, r in enumerate(results, 1):
        table.add_row(str(i), r['name'], r['current_club'], f"{r['score']:.1f}%")
    console.print(table)

    # capturando o input numérico do usuário mapeado para o item real
    idx = IntPrompt.ask("Número (0 cancela)", choices=[str(i) for i in range(len(results) + 1)])
    return None if idx == 0 else results[idx - 1]

# abstraindo a lógica de fuzzy search em clubes com seleção pelo terminal
def search_and_select_club(predictor) -> dict | None:
    query   = Prompt.ask("🔍 Clube (busca fuzzy)")
    results = predictor.search_club(query, limit=5)

    # tratando o retorno vazio para evitar erros de indexação
    if not results:
        console.print("[red]Nenhum clube encontrado.[/red]")
        return None

    # criando a interface da tabela de opções de times encontrados
    table = Table(box=box.SIMPLE, show_header=True)
    table.add_column("#",      justify="right", style="cyan",    no_wrap=True)
    table.add_column("Clube",                   style="magenta")
    table.add_column("Liga",                    style="green")
    table.add_column("Score",  justify="right", style="dim")
    for i, r in enumerate(results, 1):
        table.add_row(str(i), r['name'], str(r['league']), f"{r['score']:.1f}%")
    console.print(table)

    # transformando a escolha em índice da lista original e permitindo cancelamento
    idx = IntPrompt.ask("Número (0 cancela)", choices=[str(i) for i in range(len(results) + 1)])
    return None if idx == 0 else results[idx - 1]

# encapsulando a apresentação gráfica dos resultados do modelo 1
def show_current_value_card(player: dict, value: float):
    content = f"""
[bold magenta]Jogador:[/bold magenta]     {player['name']}
[bold green]Clube Atual:[/bold green] {player['current_club']}

[bold yellow]Valor de Mercado Estimado[/bold yellow]

  [bold white on blue]  € {value:>18,.2f}  [/bold white on blue]

[dim]Baseado nos stats mais recentes e no contexto econômico atual.[/dim]
"""
    console.print(Panel(content, title="💶 Valor de Mercado Atual", border_style="blue", expand=False))

# encapsulando a apresentação gráfica das inferências do modelo 2
def show_transfer_card(player: dict, seller: dict, buyer: dict, transfer_date: str, res: dict):
    mv  = res['market_value_eur']
    p15 = res['transfer_fee_p15_eur']
    p50 = res['transfer_fee_p50_eur']
    p85 = res['transfer_fee_p85_eur']

    # interpolando dados nas strings do painel do rich
    content = f"""
[bold magenta]Jogador:[/bold magenta]    {player['name']}
[bold red]Vendedor:[/bold red]   {seller['name']}
[bold green]Comprador:[/bold green]  {buyer['name']}
[bold blue]Data:[/bold blue]       {transfer_date}

[dim]Valor de Mercado (base do Modelo 2): € {mv:,.2f}[/dim]

[bold yellow]Estimativa de Fee de Transferência[/bold yellow]

  Pessimista  (P15)  [cyan]€ {p15:>20,.2f}[/cyan]
  Esperado    (P50)  [bold white on green]  € {p50:>20,.2f}  [/bold white on green]
  Otimista    (P85)  [cyan]€ {p85:>20,.2f}[/cyan]
"""
    console.print(Panel(content, title="🤝 Simulação de Transferência", border_style="green", expand=False))

# bloco de execução principal garantindo saída limpa em ctrl+c
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold red]Cancelado pelo usuário.[/bold red]")
        sys.exit(0)
