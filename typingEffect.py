import typer
import time

app = typer.Typer()

@app.command()
def demo_log():
    text = '''
            Food Log 03.18 - 03.20
            this file is food log :

            03.11.2023 
            xxxxxxxxxxxxxxxxxxxxxxx
            xxxxxxxxxxxxxxxxxxxxxxx

            03.12.2023
            xxxxxxxxxxxxxxxxxxxxxxx
            xxxxxxxxxxxxxxxxxxxxxxx

            03.13.2023
            xxxxxxxxxxxxxxxxxxxxxxx
            xxxxxxxxxxxxxxxxxxxxxxx

            End of the Log, Stay Healthy.    
        '''
    for char in text:
        typer.echo(char, nl=False)
        time.sleep(0.01)
    typer.echo()

if __name__ == "__main__":
    app()