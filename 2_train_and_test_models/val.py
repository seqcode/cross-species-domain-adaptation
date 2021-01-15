from utils import *
from default_params import Params
import plot_print_utils as putils
from keras.models import load_model


if __name__ == "__main__":
	params = Params(sys.argv)
	model = load_model(params.modelfile)
	print("Model loaded.")

	params.target_val_probs = val(params, model, True, True)
	putils.print_val_metrics(params, target_data = True)
	params.source_val_probs = val(params, model, False, True)
	putils.print_val_metrics(params, target_data = False)
