import rawpy
import imageio
import os
base_dir = '../data1'
dst_dir = 'data'

# Converting .nef file format to .jpg format
for i in os.listdir(base_dir):
    src = os.path.join(base_dir,i)
    dst = os.path.join(dst_dir, i)
    os.mkdir(dst)
    # print(dst)
    for f in os.listdir(src):
        file = os.path.join(src,f)
        name = f[:-4]
        # print("name: ",name)
        # print(file)
        with rawpy.imread(file) as raw:
            rgb = raw.postprocess()
            image_name = dst+"\\"+str(name)+".jpg"
            # print("image_name: ",image_name)
            imageio.imwrite(image_name, rgb)