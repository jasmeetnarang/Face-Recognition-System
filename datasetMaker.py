
# creating separate directories for gallery and probe data
import os, shutil

base_custom_dir = "face_images"
base_dir = 'data'
probe_dir = 'probe'
gallery_dir = 'gallery'

if os.path.exists(base_custom_dir) == False:
    os.mkdir(base_custom_dir)
    os.mkdir(os.path.join(base_custom_dir, gallery_dir))
    os.mkdir(os.path.join(base_custom_dir, probe_dir))

no_class = os.listdir(base_dir)

gallery_dir = os.path.join(base_custom_dir,gallery_dir)
probe_dir = os.path.join(base_custom_dir,probe_dir)
for i in range(len(no_class)):
    # Creating folders in the training set
    if os.path.exists(os.path.join(gallery_dir, no_class[i])) == False:
        os.mkdir(os.path.join(gallery_dir, no_class[i]))
    # Creating folders in validation set
    if os.path.exists(os.path.join(probe_dir, no_class[i])) == False:
        os.mkdir(os.path.join(probe_dir, no_class[i]))

# Copying the images from base_dir to our custom directory
for path in os.listdir(base_dir):
    folder = os.path.join(base_dir, path)
    count = len(os.listdir(folder))
    t = int(count/3)
    c = 0
    for i in os.listdir(folder):
        full_path = os.path.join(folder, i)
        if c <= t:
            src = os.path.join(folder, i)
            dst = os.path.join(os.path.join(probe_dir,path), i)
            shutil.copyfile(src, dst)
        elif c > t:
            src = os.path.join(folder, i)
            dst = os.path.join(os.path.join(gallery_dir,path),i)
            shutil.copyfile(src, dst)
        c += 1
