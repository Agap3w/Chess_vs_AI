"""
quick code to check if GPU is recognized.
since i'm running this on windows I had to downgrade to py3.10 and tf2.10 (with cuda 11.2 and cuDNN 8.1) and np1.23.5
alternative are running on linux, running on WSL2, a Tensorflow plugin
"""

import tensorflow as tf
print("GPU disponibile" if tf.config.list_physical_devices('GPU') else "Nessuna GPU")