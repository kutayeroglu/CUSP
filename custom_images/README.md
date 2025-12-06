Experiment guide: 
- When you want to add/change an image in custom_images and get outputs go to custom output notebook

!git clone https://github.com/kutayeroglu/CUSP
%cd CUSP



custom_images_path = "custom_images"

custom_filenames_batch = [
    os.path.join(custom_images_path, f)
    for f in next(iter(os.walk(custom_images_path)))[2]
    if f[-4:] in ['.jpg', '.png', 'jpeg']
]

# Image side
side = configs[KEY]['side']
# Read & preprocess images
imgs = preprocess_images(custom_filenames_batch, side)

# Transform to tensors
im_in_tensor = (torch.tensor(np.array(imgs))/256*2-1).cuda() # Values {-1,1}

# Aging steps
steps = 5 # N steps
# Repeat images N times
n_images = im_in_tensor.shape[0]
im_in_tensor_exp = im_in_tensor[:,None].expand([n_images,steps,*im_in_tensor.shape[1:]]).reshape([-1,*im_in_tensor.shape[1:]])
# Define target ages
labels_exp = torch.tensor(np.repeat(np.linspace(*data_labels_range,steps,dtype=int)[:,None],n_images,1).T.reshape(-1))