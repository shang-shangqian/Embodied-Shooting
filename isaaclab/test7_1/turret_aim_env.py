try:
    from .turret_aim_env_g2 import TurretAimEnv, TurretAimEnvCfg
except ImportError:
    from turret_aim_env_g2 import TurretAimEnv, TurretAimEnvCfg

__all__ = ["TurretAimEnv", "TurretAimEnvCfg"]
