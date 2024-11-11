import utils.filetools as fts
import utils.collect_imgs as ci

def test_get_max_list_folder():
    folder_path = './test/data'
    max_folder_number, folder_list = fts.get_max_list_folder(folder_path)
    print("最大数字文件夹名为:", max_folder_number)
    print("文件夹列表:", folder_list)

def test_get_max_list_image():
    folder_path = './test/data/0'
    max_image_number, image_list = fts.get_max_list_image(folder_path)
    print("最大数字图片名为:", max_image_number)
    print("图片列表:", image_list)

def test_collect_imgs():
    number_of_classes = [3, 4, 5]
    dataset_size = 30
    root_dir = './test/data'
    ci.collect_imgs(number_of_classes, dataset_size, root_dir)

if __name__ == '__main__':
    test_get_max_list_folder()
    test_get_max_list_image()
    test_collect_imgs()