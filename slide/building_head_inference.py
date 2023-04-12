#Dataset
import openslide 
from torch.utils.data import Dataset
import cv2
import numpy as np
from shapely import geometry
from fastai.vision import *
from slide.pytorch_helper import *
from slide.pytorch_augmentations import *
from queue import Queue
from tqdm import tqdm
from skimage.transform import resize


class BuildingsDataset_Head(Dataset):

    """Canine Mammary Tumor Buildings Dataset. 
    Read image and split in overlapping patches. 
    Apply model for each patch and combine results.
    Finally, apply non-maxima suppression to remove overlapping detections.

    
    create batch with patches return: (patch, x,y, size) with  function
    process_image()
    #iterate over batch in batches and get patch
    # for patch in batch:
    get image and mask 
        dict {image, gt_mask, pred_mask, coords{x, y, size}}
        image = self.get_patch(x, y)
        gt_mask = self.get_y_patch(x,y, down_factor=self.slide.level_downsamples[self._level])
        pred_mask = model(image)

        dict.append(image, gt_mask, pred_mask, coords{x,y,size})
    
    when done for slide
    gt_empty_slide_array = np.nparray(w,h,s)
    pred_empty_slide_array = np.nparray(w,h,s)

    for gt: 
    fill array with gt_masks

    got pred: 
    some kind of stitching function to combine and fill array with pred_mask




    
    Args:
        image_paths (list): list to images folder
        annotations_file (str): path to segmentation masks json database
        class_rgb_values (list): RGB values of select classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)

    New dataset_type: 
        dataset_type = „tubule_formation_segmentation“ 
        dataset_type = „pleomorphism_segmentation“
        -> use different metadata 
    
        Metadata tubule_formation_segmentation: 
            self.poly_klasse = "tub_id"
            self.name = 'supercategory_TubFormation'
            self.id = 'id_super_tub'

        Metadata pleomorphism_segmentation: 
            self.poly_klasse = "pleo_id"
            self.name = 'supercategory_Pleo'
            self.id = 'id_super_pleo'
    
    New delete tumor_id: contains tumor, because Nekrose and Tumor do not exist ist new datasets
    
    """
    
    def __init__(
            self, 
            image_paths: list, #list with image files
            annotations_file, #json file
            class_rgb_values=None,  
            transform = None,
            staining = False,
            level = None, #2,
            num_patches=None,
            width: int = 320, height: int = 320,
            white_mask_application = None, label_dict=None, 
            tumor_id = None,
            #new:
            dataset_type = None,
            overlap: int = 50,
            batch_size: int = 6,
            model: nn.Module = None,
            device: str = "cuda",
            resize = None,
            num_classes = None,
    ):
        #images folder full path
        self.num_classes = num_classes
        self.image_paths = image_paths #[Path(os.path.join(images_folder, image_id)) for image_id in sorted(os.listdir(images_folder))]
        self.overlap = overlap
        self.batch_size = batch_size
        self.model = model
        self.device = device
        self.white = 210
        self.resize = resize
        #poly_klasse and name need to be adapted for different dataset type
        self.dataset_type = dataset_type
        if dataset_type == "tubule_formation_segmentation":
            self.poly_klasse = "tub_id"
            self.name = 'supercategory_TubFormation'
            self.id = 'id_super_tub'
            self.tumor_id = None #tumor id is only needed for segmentation


        if dataset_type == "pleomorphism_segmentation":
            self.poly_klasse = "pleo_id"
            self.name = 'supercategory_Pleo'
            self.id = 'id_super_pleo'
            self.tumor_id = None #tumor id is only needed for segmentation
        
        if dataset_type == "scc_tumor_segmentation": 
            self.poly_klasse = "tumor_id"
            self.name = 'supercategory_1'
            self.id = 'id_super'
            self.tumor_id = tumor_id #tumor id is only needed for segmentation
        
        if dataset_type == "mammaCA_tumor_segmentation": 
            self.poly_klasse = "tumor_id"
            self.name = 'supercategory_1'
            self.id = 'id_super'
            self.tumor_id = tumor_id #tumor id is only needed for segmentation

        #width and height of the patch
        self.width = width
        self.height = height
        self.white_mask_application = white_mask_application
        self.label_dict = label_dict
        self.num_patches = num_patches
        self.patch_size = width
        self._level = level
        self.annotations_file = annotations_file
        self.label_dict = label_dict
        self.class_rgb_values = class_rgb_values
        self.transform = transform 
        self.staining = staining
        
    # # neu
    @property
    def level(self):
        return self._level


    def get_patch(self, x: int = 0, y: int = 0, size: tuple = (512, 512)):
        x = x
        y = y
        rgb = np.array(self.slide.read_region(location= ((int(x)), (int(y))),
                                              level=self._level, size=size))[:, :, :3]
        return rgb
        # Funktion um die Konturen zu zeichnen
    def get_y_patch(self, x: int = 0, y: int = 0,size: tuple = (512, 512), down_factor: int = None):
        y_patch = np.zeros(shape=(size[1], size[0]), dtype=np.int8)
        inv_map = {v: k for k, v in self.tissue_classes.items()}  

        contains_tumor = False

        # über die Polygone iterieren - eine Liste mit Json Formatierung der Annotationen 
        for poly in self.polygons:
            coordinates = np.array(poly['segmentation']).reshape(
                (-1, 2)) // down_factor
            # coordinates für patch anpassen
            coordinates = coordinates - (x, y)
            item = poly[self.poly_klasse]
            
            #test whether contains tumor: 
            label = self.label_dict[inv_map[item]]
            
            #Truth value, whether poly contains tumor - only for mamma_ca
            if self.tumor_id != None and poly["category_id"] == self.tumor_id:  
                contains_tumor = True

            y_patch = cv2.drawContours(y_patch, [coordinates.reshape(
                (-1, 1, 2)).astype(int)], -1, label, -1)
        
        #if true draw tumor as dominant label - GEÄNDERT
        if contains_tumor:
            #only draw tumor contours 
            for poly in self.polygons: 
                item = poly[self.poly_klasse]
                label = self.label_dict[inv_map[item]]
                coordinates = np.array(poly['segmentation']).reshape(
                (-1, 2)) // down_factor
                coordinates = coordinates - (x, y)

                if poly["category_id"] == 2:
                    cv2.drawContours(y_patch, [coordinates.reshape(
                    (-1, 1, 2)).astype(int)], -1, label, -1)

        # white mask for parts without annotation
        if self.white_mask_application:
            #white mask with true or false values
            white_mask = cv2.cvtColor(self.get_patch(
                x, y, (size[0], size[1])), cv2.COLOR_RGB2GRAY) > self.white

            y_patch[white_mask] = 1
        
                     
        return y_patch
    
    
    def get_coordinates_queue(self, slide_w: int, slide_h: int, patch_size: int, overlap: int):
        """
        creates queue with overlapping patch coordinates for the whole image
        """
        coords_queue = Queue()
        num_patches = 0
        for x in np.arange(0, slide_w, patch_size-overlap): 
            for y in np.arange(0, slide_h, patch_size-overlap): 
                # avoid black borders
                if x + patch_size > slide_w:
                    x = slide_w - patch_size
                if y + patch_size > slide_h:
                    y = slide_h - patch_size
                coords_queue.put((x,y))
                num_patches += 1

        return coords_queue, num_patches
    
    def get_batch_from_queue(self,
            queue: Queue, 
            batch_size: int,
            ):
        """Create batch of patches from the queue and convert to tensor.
        """
        batch_images = []
        batch_x = []
        batch_y = []
        for _ in range(batch_size):
            if queue.qsize() > 0:
                # get coords from queue
                x, y = queue.get()
                # load patch
                patch  = self.get_patch(int(x), int(y), (self.width, self.height))
                #preprocess 
                #apply staining 
                if self.staining: 
                    patch = get_stained_patch(patch,patch_size = self.patch_size, num_patches = self.num_patches)
                
                if self.transform:
                    augmented = self.transform(image = patch)
                    patch = augmented['image']
                #convert to tensor
                patch = torch.from_numpy(patch).permute(2, 0, 1).type(torch.float32)
                # add to batch
                batch_images += [patch]
                batch_x += [int(x)]
                batch_y += [int(y)]
            else: 
                break
        return batch_images, batch_x, batch_y

    #function for tubule segmentation
    def get_areas(self, full_pred_mask_array: array = None):
        areas = {}
        slide_area = self.slide_dim_0 * self.slide_dim_1
        #calculate percentage of classes in mask
        for label, label_index in self.label_dict.items(): 
            #sum up label area in empty_array 
            area_sum = np.sum(full_pred_mask_array == label_index)
            areas[label] = round(area_sum/slide_area, 2)
        return areas
    
    #function for pleomorphism
    def get_dominant_class(self, full_pred_mask_array: array = None):
        #get unique values from mask as list
        mask_values = np.unique(full_pred_mask_array)
        #get max value from list 
        max_value = max(mask_values)
        if max_value == 2: 
            return 1
        if max_value == 3: 
            return 2
        if max_value == 4: 
            return 3

    


    #load patches for image and preprocess
    def __getitem__(self, i): #i is type dict
        with open(self.annotations_file) as f:
            data = json.load(f) 
            #all classes dict
            self.tissue_classes = dict(zip([cat[self.name] for cat in data["categories"]], [
                                       cat[self.id] for cat in data["categories"]]))

            #get image filename
            filename = self.image_paths[i]

            #from full path get only name
            name = Path(filename).name
            image_id = [i.get('id') for i in data.get("images") if i.get("file_name") == name][0]

            #get annotations for image id
            self.polygons = [anno for anno in data['annotations'] if anno["image_id"] == image_id]
        
        
        #read slide
        self.slide = openslide.open_slide(str(filename))
        self.down_factor = self.slide.level_downsamples[self._level]
        #width and height of the slide
        self.slide_dim_0 = int(self.slide.dimensions[0]//self.down_factor)
        self.slide_dim_1 = int(self.slide.dimensions[1]//self.down_factor)
        
        #get patch coordinates from batch with patch creation function 
        coords_queue, num_patches = self.get_coordinates_queue(self.slide_dim_0, self.slide_dim_1,self.patch_size, self.overlap)
        num_batches = int(np.ceil(num_patches / self.batch_size))

        #initialize results TODO : get full gt_mask and image 
        res = {'image': [], 'gt_mask': [], 'pred_masks': [], 'coords': []} #coords{x,y,size}

        #make empty np array in image size
        full_pred_mask_array = np.zeros((self.slide_dim_1, self.slide_dim_0)) #height, width

        #loop over batches
        for _ in tqdm(range(num_batches), desc= 'Processing image'): 
            #get batch - gets normalized here
            batch_images, batch_x, batch_y = self.get_batch_from_queue(coords_queue, self.batch_size)
            #push to device
            batch_images = [p.to(self.device) for p in batch_images]

            # apply model
            with torch.no_grad():
                pred_masks = []
                #append predictions to masks list
                for batch_img in batch_images:
                    #resize 
                    if self.resize:
                        batch_img = batch_img.reshape(1,3,self.patch_size,self.patch_size)
                        batch_img = batch_img[:, :, ::2, ::2]
                        pred_mask = self.model(batch_img) #1,4,512,512 ist die shape vom output 
                        #back to original size 
                        
                        #back to cpu
                        pred_mask = pred_mask.detach().squeeze().cpu().numpy()
                        pred_mask = resize(pred_mask, (self.num_classes,1024, 1024))
                    else: 
                        batch_img = batch_img.reshape(1,3,self.patch_size,self.patch_size)
                        batch_img = batch_img[:, :, ::2, ::2]
                        pred_mask = self.model(batch_img) #1,4,512,512 ist die shape vom output 
                        
                        #back to cpu
                        pred_mask = pred_mask.detach().squeeze().cpu().numpy()
                        # pred_mask = resize(pred_mask, (4,1024, 1024))
                    pred_masks.append(pred_mask) #mask with probabilities 
            
            #add results from pred masks to empty array
            for mask, x_orig, y_orig in zip(pred_masks, batch_x, batch_y): 
                
                res['pred_masks'] += [mask]
                res['coords'] += [x_orig, y_orig]

                mask = np.transpose(mask,(1,2,0))
                #convert decoded class dim into 1 encoded dim
                mask = reverse_one_hot(mask) 
                red = 0 
                min_x_cut = red
                max_x_cut = red
                min_y_cut = red
                max_y_cut = red
                

                if x_orig == 0: 
                    min_x_cut = 0
                if x_orig == full_pred_mask_array.shape[0]: 
                    max_x_cut = 0
                if y_orig == 0: 
                    min_y_cut = 0
                if y_orig == full_pred_mask_array.shape[1]: 
                    max_y_cut = 0
                mask = mask[min_y_cut:self.patch_size-max_y_cut, min_x_cut:self.patch_size-max_x_cut]
                s = x_orig + min_x_cut
                e = x_orig + min_x_cut + mask.shape[0]
                s_2 = y_orig + min_y_cut
                e_2 = y_orig+ min_y_cut + mask.shape[1]
                # full_pred_mask_array[(x_orig + min_x_cut):(x_orig + min_x_cut + mask.shape[0]), (y_orig + min_y_cut): (y_orig+ min_y_cut + mask.shape[1])] = mask
                full_pred_mask_array[s_2: e_2, s:e] = mask
                
        
        #call function to display tubule areas in percent 
        if self.dataset_type == "tubule_formation_segmentation":
            areas = self.get_areas(full_pred_mask_array)
            print("predicted areas in percent", areas)
        
        #call function to display pleomorphism grade
        if self.dataset_type == "pleomorphism_segmentation":
            grade = self.get_dominant_class(full_pred_mask_array)
            print("pleomorphism grades", grade)
            
        #load full image for return -- w,h dimension
        image = self.get_patch(0,0, (self.slide_dim_0, self.slide_dim_1))
        #load full gt_mask 
        gt_mask = self.get_y_patch(0,0,(self.slide_dim_0, self.slide_dim_1), self.down_factor)

        print("mask shape", full_pred_mask_array.shape)
        print("image shape", image.shape)

        if self.dataset_type == "pleomorphism_segmentation":
            return [image, gt_mask, full_pred_mask_array, grade]

        #return image and mask
        if self.dataset_type == "tubule_formation_segmentation":
            return [image, gt_mask, full_pred_mask_array, areas]
        else:
            return [image, gt_mask, full_pred_mask_array]
    
        #TODO: stitch together masks 


    def __len__(self):
        # return length of 
        return len(self.image_paths)
