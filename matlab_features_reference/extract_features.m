%% vgg / caffe spec

use_gpu = 0;
gpu_id = 1;

% Set caffe mode
if use_gpu
  caffe.set_mode_gpu();
  caffe.set_device(gpu_id);
else
  caffe.set_mode_cpu();
end

model = './models/vgg_ilsvrc_16/VGG_ILSVRC_16_layers_deploy.prototxt';
weights = './models/vgg_ilsvrc_16/VGG_ILSVRC_16_layers.caffemodel';
batch_size = 10;

net = caffe.Net(model, weights, 'test');

%% input files spec

root_path = './data/flickr30k/';
fs = textread([root_path 'all_imgs.txt'], '%s');
N = length(fs);

%%

% iterate over the images in batches
feats = zeros(4096, N, 'single');
for b=1:batch_size:N

    % enter images, and dont go out of bounds
    Is = {};
    for i = b:min(N,b+batch_size-1)
        I = imread([root_path fs{i}]);
        if ndims(I) == 2
            I = cat(3, I, I, I); % handle grayscale edge case. Annoying!
        end
        Is{end+1} = I;
    end
    input_data = prepare_images_batch(Is);

    tic;
    scores = net.forward({input_data});
    scores = squeeze(scores{1});
    tt = toc;

    nb = length(Is);
    feats(:, b:b+nb-1) = scores(:,1:nb);
    fprintf('%d/%d = %.2f%% done in %.2fs\n', b, N, 100*(b-1)/N, tt);
end

%% write to file

save([root_path 'vgg_feats_hdf5.mat'], 'feats', '-v7.3');
save([root_path 'vgg_feats.mat'], 'feats');
