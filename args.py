
class Args():

	# training args
	path_ir = "images/new_dataests/IR"
	# path_ir = 'images/train_images_20100/IR'
	save_model_dir = "models"  # "path to folder where trained model will be saved."
	save_loss_dir = "models/loss"  # "path to folder where trained model will be saved."
	vgg_model_dir = 'models/vgg19'
	lr = 0.00001
	batch_size = 3  # "batch size for training, default is 4"
	epochs = 4  # "number of training epochs, default is 4"
	n = 128  # number of filters
	s = 3  # filter size
	channel = 1  # 1 - gray, 3 - RGB
	stride = 1
	resume = None
	cuda = 1  # "set it to 1 for running on GPU, 0 for CPU"
	Height = 256
	Width = 256

	# Y2:
	lam2_vi_list = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5]
	lam2_vi = lam2_vi_list[0]
	# Wir:
	w_ir_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
	w_ir = w_ir_list[4]
	# Y4:
	lam3_gram_list = [1000, 1500, 2000, 2500, 3000]
	lam3_gram = lam3_gram_list[2]





