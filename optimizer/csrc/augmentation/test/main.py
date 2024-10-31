from augmentation import Transform, open_image, save_png

img = open_image('test/input.png')
transform = Transform()
transform.set_infill_sizes(16, 64, 16, 64)
transform.set_noise_levels(16, 64)
result = transform(img)
save_png('result.png', result)

