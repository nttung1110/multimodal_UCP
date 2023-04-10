from src.utils import Config
from .base_ucp import BaseUCP

def get_ucp(config: Config):
    ucp = {
        'base': BaseUCP
    }

    return ucp[config.ucp.name](config.ucp)