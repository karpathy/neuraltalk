function images = prepare_images_batch(im_cell)
% takes a cell array of images in raw uint8 form

mean_pix = [103.939, 116.779, 123.68]; % VGG mean assumed
batch_size = 10;

CROPPED_DIM = 224;

% insert images
images = zeros(CROPPED_DIM, CROPPED_DIM, 3, batch_size, 'single');
N = length(im_cell);
for i=1:N

    % resize to fixed input
    im = single(im_cell{i});
    im = imresize(im, [CROPPED_DIM CROPPED_DIM]);

    % RGB -> BGR
    im = im(:, :, [3 2 1]);

    % flip xy
    images(:, :, :, i) = permute(im, [2, 1, 3]);
end

% mean BGR pixel subtraction
for c = 1:3
    images(:, :, c, :) = images(:, :, c, :) - mean_pix(c);
end
