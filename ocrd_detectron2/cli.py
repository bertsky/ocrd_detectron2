import click

from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from .segment import Detectron2Segment

@click.command()
@ocrd_cli_options
def ocrd_detectron2_segment(*args, **kwargs):
    return ocrd_cli_wrap_processor(Detectron2Segment, *args, **kwargs)
