import torch.utils.data as data

from PIL import Image
import os
import os.path

from sklearn.model_selection import train_test_split
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):

    classes = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]
    image = [img for img in classes if img.find('frames_cleanpass') > -1]
    disp  = [dsp for dsp in classes if dsp.find('disparity') > -1]

    all_left_img=[]
    all_right_img=[]
    all_left_disp = []
    test_left_img=[]
    test_right_img=[]
    test_left_disp = []

    '''monkaa_path = filepath + [x for x in image if 'monkaa' in x][0]
    monkaa_disp = filepath + [x for x in disp if 'monkaa' in x][0]

    monkaa_dir = os.listdir(monkaa_path)

    for dd in monkaa_dir:
      for im in os.listdir(monkaa_path+'/'+dd+'/left/'):
        if is_image_file(monkaa_path+'/'+dd+'/left/'+im):
          all_left_img.append(monkaa_path+'/'+dd+'/left/'+im)
        all_left_disp.append(monkaa_disp+'/'+dd+'/left/'+im.split(".")[0]+'.pfm')

      for im in os.listdir(monkaa_path+'/'+dd+'/right/'):
        if is_image_file(monkaa_path+'/'+dd+'/right/'+im):
          all_right_img.append(monkaa_path+'/'+dd+'/right/'+im)

    flying_path = filepath + [x for x in image if x == 'frames_cleanpass'][0]
    flying_disp = filepath + [x for x in disp if x == 'frames_disparity'][0]
    flying_dir = flying_path+'/TRAIN/'
    subdir = ['A','B','C']

    for ss in subdir:
      flying = os.listdir(flying_dir+ss)

      for ff in flying:
        imm_l = os.listdir(flying_dir+ss+'/'+ff+'/left/')
        for im in imm_l:
          if is_image_file(flying_dir+ss+'/'+ff+'/left/'+im):
            all_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)
          
          all_left_disp.append(flying_disp+'/TRAIN/'+ss+'/'+ff+'/left/'+im.split(".")[0]+'.pfm')

          if is_image_file(flying_dir+ss+'/'+ff+'/right/'+im):
            all_right_img.append(flying_dir+ss+'/'+ff+'/right/'+im)

    flying_dir = flying_path+'/TEST/'

    subdir = ['A','B','C']

    for ss in subdir:
      flying = os.listdir(flying_dir+ss)

      for ff in flying:
        imm_l = os.listdir(flying_dir+ss+'/'+ff+'/left/')
        for im in imm_l:
          if is_image_file(flying_dir+ss+'/'+ff+'/left/'+im):
            test_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)
          
          test_left_disp.append(flying_disp+'/TEST/'+ss+'/'+ff+'/left/'+im.split(".")[0]+'.pfm')

          if is_image_file(flying_dir+ss+'/'+ff+'/right/'+im):
            test_right_img.append(flying_dir+ss+'/'+ff+'/right/'+im)'''



    driving_dir = filepath + [x for x in image if 'driving' in x][0] + '/'
    driving_disp = filepath + [x for x in disp if 'driving' in x][0]

    subdir1 = ['35mm_focallength','15mm_focallength']
    subdir2 = ['scene_backwards','scene_forwards']
    subdir3 = ['fast','slow']

    for i in subdir1:
      for j in subdir2:
        for k in subdir3:
            imm_l = sorted(os.listdir(driving_dir+i+'/'+j+'/'+k+'/left/'))
            for im in imm_l:
              if is_image_file(driving_dir+i+'/'+j+'/'+k+'/left/'+im):
                all_left_img.append(driving_dir+i+'/'+j+'/'+k+'/left/'+im)

              all_left_disp.append(driving_disp+'/'+i+'/'+j+'/'+k+'/left/'+im.split(".")[0]+'.pfm')

              if is_image_file(driving_dir+i+'/'+j+'/'+k+'/right/'+im):
                all_right_img.append(driving_dir+i+'/'+j+'/'+k+'/right/'+im)


    # driving이 8800개 이미지, 4400개 이미지 쌍, batch_size=4 기준 1100회 Iter, time=4.10 기준 4510s = 약 75.17분 소요
    
    # 이미지 뽑기 전, left/right 이미지 쌍으로 합쳐주기
    all_img = np.array([all_left_img, all_right_img]).T

    all_data_num = len(all_img)

    # (driving 기준) 4400개 쌍에서 얼마나 사용할지 비율
    all_ratio = 1.
    # 전체 사용할 것 아니면, testset 뽑듯이 일부만 뽑아내기
    if all_ratio < 1.:
      _, all_img, _, all_left_disp = train_test_split(all_img, all_left_disp, test_size = all_ratio, shuffle = True, random_state = 1)

    use_data_num = len(all_img)

    # test에 얼마나 사용할지 비율 (위에서 뽑은 것 중.)
    test_ratio = .2
    img_train, img_test, all_left_disp, test_left_disp = train_test_split(all_img, all_left_disp, test_size = test_ratio, shuffle = True, random_state = 1)

    # 쌍으로 합쳐줬던 이미지 left/right 분리
    img_train = img_train.T
    all_left_img = img_train[0].tolist()
    all_right_img = img_train[1].tolist()
    train_data_num = len(all_left_img)
    
    img_test = img_test.T
    test_left_img = img_test[0].tolist()
    test_right_img = img_test[1].tolist()
    test_data_num = len(test_left_img)

    print("전체 data 수:", all_data_num)
    print("사용할 data 수: %d (전체 data의 %.2f%%)" % (use_data_num, use_data_num/all_data_num*100))
    print("Train data 수: %d (사용할 data의 %.2f%%)" %(train_data_num, train_data_num/use_data_num*100))
    print("Test data 수: %d (사용할 data의 %.2f%%)" %(test_data_num, test_data_num/use_data_num*100))

    return all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp