import caffe
import numpy as np
import argparse
import os
import time
import scipy.io


def reduce_along_dim(img , dim , weights , indicies): 
    other_dim = abs(dim-1)       
    if other_dim == 0:  #resizing image width
        weights  = np.tile(weights[np.newaxis,:,:,np.newaxis],(img.shape[other_dim],1,1,3))
        out_img = img[:,indicies,:]*weights
        out_img = np.sum(out_img,axis=2)
    else:   # resize image height     
        weights  = np.tile(weights[:,:,np.newaxis,np.newaxis],(1,1,img.shape[other_dim],3))
        out_img = img[indicies,:,:]*weights
        out_img = np.sum(out_img,axis=1)
        
    out_img = np.clip(out_img,0,255)
    return out_img

            
def cubic_spline(x):
    absx   = np.abs(x)
    absx2  = absx**2
    absx3  = absx**3 
    kernel_weight = (1.5*absx3 - 2.5*absx2 + 1) * (absx<=1) + (-0.5*absx3 + 2.5* absx2 - 4*absx + 2) * ((1<absx) & (absx<=2))
    return kernel_weight
    
def contribution(in_dim_len , out_dim_len , scale ):
    if scale < 1:
        kernel_width =  4 / scale
        
    x_out = np.array(range(1,out_dim_len+1))  
    #project to the input space dimension
    u = x_out/scale + 0.5*(1-1/scale)
    
    #position of the left most pixel in each calculation
    l = np.floor( u - kernel_width/2)
  
    #maxium number of pixels in each computation
    p = int(np.ceil(kernel_width) + 2)
    
    indicies = np.zeros((l.shape[0],p) , dtype = int)
    indicies[:,0] = l
      
    for i in range(1,p):
        indicies[:,i] = indicies[:,i-1]+1
    
    #compute the weights of the vectors
    u = u.reshape((u.shape[0],1))
    u = np.repeat(u,p,axis=1)
    
    if scale < 1:
        weights = scale*cubic_spline(scale*(indicies-u ))
    else:
        weights = cubic_spline((indicies-u))
         
    weights_sums = np.sum(weights,1)
    weights = weights/ weights_sums[:, np.newaxis] 
    
    indicies = indicies - 1    
    indicies[indicies<0] = 0                     
    indicies[indicies>in_dim_len-1] = in_dim_len-1 #clamping the indicies at the ends
    
    valid_cols = np.all( weights==0 , axis = 0 ) == False #find columns that are not all zeros
    
    indicies  = indicies[:,valid_cols]           
    weights    = weights[:,valid_cols]
    
    #round weights to 4 decimal place since matlab precision is limited 4 decimal place (Matlab2010)
    weights = np.round(weights,4)
    return weights , indicies
     
def imresize(img , cropped_width , cropped_height):
    width_scale  = float(cropped_width)  / img.shape[1]
    height_scale = float(cropped_height) / img.shape[0] 
    
    order   = np.argsort([height_scale , width_scale])
    scale   = [height_scale , width_scale]
    out_dim = [cropped_height , cropped_width] 
    
    
    weights  = [0,0]
    indicies = [0,0]
    for i in range(0 , 2):
        weights[i] , indicies[i] = contribution(img.shape[ i ],out_dim[i], scale[i])
    
    for i in range(0 , len(order)):
        img = reduce_along_dim(img , order[i] , weights[order[i]] , indicies[order[i]])
        
    return img.astype(np.uint8)

        
def caffe_extract_feats(path_imgs , path_model_def , path_model , batch_size = 1 , WITH_GPU = True):
    '''
    Function using the caffe python wrapper to extract 4096 from VGG_ILSVRC_16_layers.caffemodel model
    
    Inputs:
    ------
    path_imgs      : list of the full path of images to be processed 
    path_model_def : path to the model definition file
    path_model     : path to the pretrained model weight
    WItH_GPU       : Use a GPU 
    
    Output:
    -------
    features           : return the features extracted 
    '''
    
    if WITH_GPU:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    print "loading model:",path_model
    caffe_net = caffe.Classifier(path_model_def , path_model , image_dims = (224,224) , raw_scale = 255,
                            mean = np.array([103.939, 116.779, 123.68]) )

    feats = np.zeros((4096 , len(path_imgs)))
    
    for b in range(0 , len(path_imgs) , batch_size):
        list_imgs = []
        for i in range(b , b + batch_size ):
            if i < len(path_imgs):
                list_imgs.append( np.array(caffe.io.load_image(path_imgs[i]) ) )
            else:
                list_imgs.append(list_imgs[-1]) #Appending the last image in order to have a batch of size 10. The extra predictions are removed later..
                
        caffe_input = np.asarray([caffe_net.transformer.preprocess('data', in_) for in_ in list_imgs]) #preprocess the images
       
        predictions =caffe_net.forward(data = caffe_input)
        predictions = predictions[caffe_net.outputs[0]].transpose()
        
        if i < len(path_imgs):
            feats[:,b:i+1] = predictions
            n = i+1
        else:
            n = min(batch_size , len(path_imgs) - b) 
            feats[:,b:b+n] = predictions[:,0:n] #Removing extra predictions, due to the extra last image appending.
            n += b 
        print "%d out of %d done....."%(n ,len(path_imgs))

    return feats      
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_def_path',dest='model_def_path', type=str , help='Path to the VGG_ILSVRC_16_layers model definition file.')
    parser.add_argument('--model_path', dest='model_path',type=str,  help='Path to VGG_ILSVRC_16_layers pretrained model weight file i.e VGG_ILSVRC_16_layers.caffemodel')
    parser.add_argument('-i',dest='input_directory',help='Path to Directory containing images to be processed.')
    parser.add_argument('--filter',default = None ,dest='filter', help='Text file containing images names in the input directory to be processed. If no argument provided all images are processed.')
    parser.add_argument('--WITH_GPU',default = True , type = bool , dest='WITH_GPU', help = 'Caffe uses GPU for feature extraction')

    args = parser.parse_args()
    
    input_directory = args.input_directory
    path_model_def_file = args.model_def_path
    path_model  = args.model_path
    filter_path = args.filter
    WITH_GPU    = args.WITH_GPU
    
    if not os.path.exists(input_directory):
        raise RuntimeError("%s , Directory does not exist"%(input_directory))
    
    if not os.path.exists(path_model_def_file):
        raise RuntimeError("%s , Model definition file does not exist"%(path_model_def_file))
    
    if not os.path.exists(path_model):
        raise RuntimeError("%s , Path to pretrained model file does not exist"%(path_model))
    
    if not filter_path == None:
        imgs = open(filter_path,'r').read().splitlines()        
    else:
        imgs = os.listdir(input_directory)
    
    path_imgs = [ os.path.join(input_directory , file) for file in imgs ]
    
    start_time = time.time()
    print "Feature Extraction for %d images starting now"%(len(path_imgs))
    feats = caffe_extract_feats(path_imgs, path_model_def_file, path_model, WITH_GPU)
    print "Total Duration for generating predictions %.2f seconds"%(time.time()-start_time)
    
    out_path = os.path.join(input_directory,'vgg_feats.mat')
    print "Saving prediction to disk %s"%(out_path)
    vgg_feats = {}
    vgg_feats['feats'] = feats
    
    scipy.io.savemat(out_path , vgg_feats)
    
    print "Have a Good day!"
    
    
    
    
   