from utils import *
from DA_utils import *
from DA_params import DA_Params
import plot_print_utils as putils
from keras.models import load_model
from flipGradientTF import GradientReversal

if __name__ == "__main__":
	params = DA_Params(sys.argv)
	
	model = load_model(params.modelfile, custom_objects={"GradientReversal":GradientReversal, "custom_loss":custom_loss})
	print("Model loaded.")

	params.target_val_probs = val(params, model, True, True)
	putils.print_val_metrics(params, target_data = True)
	params.source_val_probs = val(params, model, False, True)
	putils.print_val_metrics(params, target_data = False)
