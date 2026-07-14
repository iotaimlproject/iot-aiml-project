import logging

logger = logging.getLogger(__name__)


def configure_gpu():
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU enabled: {gpu.name} ({gpu.device_type})")
            except RuntimeError as e:
                logger.warning(f"GPU memory growth config failed: {e}")
        logger.info(f"Total GPUs available: {len(gpus)}")
    else:
        logger.info("No GPU detected — falling back to CPU")

    return len(gpus)
