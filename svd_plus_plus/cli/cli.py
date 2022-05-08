import click

from svd_plus_plus.cli.options import cli_group


@cli_group(invoke_without_command=True, context_settings={"help_option_names": ["-h", "--help"]})
@click.pass_context
def cli(ctx: click.Context) -> None:
    if ctx.invoked_subcommand is None:
        ctx.invoke(help)


@cli.command(help="Display formatted help message.")
@click.pass_context
def help(ctx: click.Context) -> None:
    click.echo(ctx.parent.get_help())


def get_cli():
    from svd_plus_plus.cli.commands import prepare_data, train

    cli.add_command(train)
    cli.add_command(prepare_data)
    return cli()


if __name__ == "__main__":
    get_cli()
