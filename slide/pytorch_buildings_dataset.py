#Dataset
import openslide 
from torch.utils.data import Dataset
from cv2 import drawContours, cvtColor, COLOR_RGB2GRAY
import numpy as np
from shapely import geometry
from fastai.vision import random, json, Path
from slide.pytorch_helper import one_hot_encode
from slide.pytorch_augmentations import get_stained_patch
from skimage.transform import resize


class BuildingsDataset(Dataset):

    """Canine Mammary Tumor Buildings Dataset. Read images, apply augmentation and preprocessing transformations.
    
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
    
    """
    
    def __init__(
            self, 
            image_paths: list, #list with image files
            annotations_file, #json file
            num_patches,
            class_rgb_values=None,  
            label_dict = None,
            reduced_label_dict=None, 
            patch_size: int = 320, 
            transform = False,
            staining = False,
            level = None, #2,
            white_mask_application = None, 
            poly_klasse = None, 
            tumor_id = None,
            slide_resize = None,
            label_dis: list = False,
            
    ):
        self.image_paths = image_paths
        self.num_patches = num_patches
        self.class_rgb_values = class_rgb_values
        self.reduced_label_dict = reduced_label_dict
        self.transform = transform
        self.staining = staining
        self.width = patch_size
        self.height = patch_size
        self.patch_size = patch_size
        self.white_mask_application = white_mask_application
        self.label_dict = reduced_label_dict
        self.poly_klasse = poly_klasse
        self.tumor_id = tumor_id
        self.level = level
        self.tissue_classes = label_dict
        self.data = self.load_data(annotations_file)
        self.slide_objects = self.load_slide_objects(slide_resize)
        slide = openslide.open_slide(str(image_paths[0]))
        self.down_factor = slide.level_downsamples[self.level]
        self.patches = self.sample_patches(label_dis=label_dis)
        self.white = 210


    def load_data(self, annotations_file):
        """Load and Initialize Json Dataframe
        Returns: 
            Dataframe
        """
        with open(annotations_file) as f: 
            #return data
            return json.load(f) 
    
    def load_slide_objects(self, slide_resize): 
        """Initialize slide objects from dataframe with the help of image paths
        
        Retuturns: 
            Dict[filename: Dict[SlideObject,PolygonsObject, List[Labels]]
        """
        slide_objects = {}
        #iterate over image paths (do i need the label here ? guess not)
        for filename in self.image_paths: 
            #from full path get only name and id
            name = Path(filename).name
            image_id = [i.get('id') for i in self.data.get("images") if i.get("file_name") == name][0]
            #get polygons object contains: area, segmentation, bbox, labels
            polygons = [anno for anno in self.data['annotations'] if anno["image_id"] == image_id]
            labels = list(dict.fromkeys([i.get(self.poly_klasse) for i in self.data.get("annotations") if i.get("image_id") == image_id])) #use dict to clean up dublicates
            if 0 in labels: 
                labels.remove(0)
            #sometimes classes are not in slide then dont put in slide objects
            if len(labels) > 0:
                #get slide object
                slide = openslide.open_slide(str(filename))
                if slide_resize == True:
                    slide = resize(slide[-1], (512, 512))
                #append to Dict[filename: List[SlideObject,PolygonsObject]
                slide_objects[name] = {"slide": slide, "polygons": polygons, "labels": labels}

        
        return slide_objects

    def sample_patches(self, label_dis: list): 
        """sample patches from all slides (default: same probability). 
        Choose label from reduced_label_dict and look for slide that contains label. 
        Pass slide and label to get_coordinates() function, to get coordinates.

        Returns: 
            Dict[str: coords]: Dictionary with {idx: {"coords": (x,y), "slide": slide, "polygons": polygons}}    
        """
        patches = {}
        #iterate over files: 
        idx = 0
        while idx < (self.num_patches*len(self.image_paths)):
            #initialize weights before
            label = random.choices(list(self.reduced_label_dict.values()),weights=label_dis, k=1)[0]
            #find slide from slide objects that contains label in label list
            # print([dict for dict in self.slide_objects if label in dict["labels"]])
            filenames = [filename for filename, dict in self.slide_objects.items() if label in dict["labels"]]
            
            # img_ids = [dict["image_id"] for dict in self.data["annotations"] if dict[self.poly_klasse]== label]
            #choose random id
            filename = random.choice(filenames)
            #look for filename with that id, so that i can look for corresponding slide from load_slide_obj
            # filename = [dict["file_name"] for dict in self.data["images"] if dict["id"] == img_id][0]
            if str(filename) in self.slide_objects:
                #keys are slide, polygons, labels
                slide,polygons,_ = self.slide_objects[filename].values()
                #check if polygons contains label
                polys_with_label = [poly for poly in polygons if poly[self.poly_klasse] == label]
                #get coords to generate patch from slide
                x,y = self.get_new_train_coordinates(slide_w= slide.dimensions[0], slide_h= slide.dimensions[1], polygons= polys_with_label)
                
            
                patches[idx] = {"coords": (x,y), "slide": slide, "polygons": polygons}
                idx += 1
        
        return patches

    
    def get_patch(self, x: int = 0, y: int = 0, slide: object = None): #TODO: slide als argument übergeben 
        rgb = np.array(slide.read_region(location=(int(x * self.down_factor), int(y * self.down_factor)),
                                              level=self.level, size=(self.width, self.height)))[:, :, :3]
        return rgb
        # Funktion um die Konturen zu zeichnen
    def get_y_patch(self, x: int = 0, y: int = 0, slide: object = None, polygons: list = None):
        y_patch = np.zeros(shape=(self.height, self.width), dtype=np.int8)
        inv_map = {v: k for k, v in self.tissue_classes.items()}  
        
        #prüfen ob Tumor Label enthalten und wenn ja, dann Nekrosen bzw. Nicht Tumor innnerhalb von Tumor überschreiben 
        contains_tumor = False

        # iterate over polygons
        for poly in polygons: 
            coordinates = np.array(poly['segmentation']).reshape(
                (-1, 2)) / self.down_factor
            # coordinates für patch anpassen
            coordinates = coordinates - (x, y)
            item = poly[self.poly_klasse]
            
            #test whether contains tumor: 
            label = self.tissue_classes[inv_map[item]]
            
            #Truth value, whether poly contains tumor - only for mamma_ca
            if self.tumor_id != None and poly["category_id"] == self.tumor_id:  
                contains_tumor = True

            y_patch = drawContours(y_patch, [coordinates.reshape(
                (-1, 1, 2)).astype(int)], -1, label, -1)
        
        #if true draw tumor as dominant label 
        if contains_tumor:
            #only draw tumor contours 
            for poly in polygons:
                item = poly[self.poly_klasse]
                label = self.tissue_classes[inv_map[item]]
                coordinates = np.array(poly['segmentation']).reshape(
                (-1, 2)) / self.down_factor
                coordinates = coordinates - (x, y)

                if poly["category_id"] == 2:
                    drawContours(y_patch, [coordinates.reshape(
                    (-1, 1, 2)).astype(int)], -1, label, -1)

        # white mask for parts without annotation 
        if self.white_mask_application:
            #white mask with true or false values
            white_mask = cvtColor(self.get_patch( 
                x, y, slide), COLOR_RGB2GRAY) > self.white
            # white mask, patch und y_patch zusammen
            y_patch[white_mask] = 1
                     
        return y_patch
    
    def get_new_train_coordinates(self, slide_w, slide_h, polygons):
        # inv_map = {v: k for k, v in self.tissue_classes.items()}
        xmin, ymin = 0,0
        slide_w = slide_w//self.down_factor
        slide_h = slide_h//self.down_factor
        found = False
        last_round = False

        while not found:
            iter = 0
            
            #in polygons choose random area and get bbox
            polygons_area = [poly["area"] for poly in polygons]
            polygons_area = np.array(polygons_area) / sum(polygons_area)
            # polygons area serves as weights, if bigger than rather take that
            polygon = random.choices(polygons, polygons_area)[0]
            coordinates = np.array(polygon['segmentation']).reshape((-1, 2))
            minx, miny = coordinates[0]
            xrange, yrange = coordinates[-1]
            while iter < 13 and not found:
                iter += 1

                if iter < 13:
                    #search for random point
                    pnt = geometry.Point(random.uniform(minx, xrange), random.uniform(miny, yrange))
                else: 
                    #take point from the middle of coordinates
                    length = len(coordinates)
                    middle = length//2
                    x,y = coordinates[middle]
                    pnt = geometry.Point(x,y)

                    last_round = True
                #check whether it is in polygon if not iterate again
                if geometry.Polygon(coordinates).contains(pnt) or last_round == True:
                    found = True
                    #get coords in middle 
                    xmin = pnt.x // self.down_factor - self.width/ 2
                    ymin = pnt.y // self.down_factor - self.height/2

                    if (xmin+self.width > slide_w):
                        #subsract the difference
                        xmin = xmin-((xmin+self.width)-slide_w)

                    if (xmin < 0): 
                        xmin = 0

                    if (ymin+self.height > (slide_h)):
                        #subsract the difference
                        ymin = ymin-((ymin+self.height)-slide_h)
                    
                    if (ymin < 0): 
                        ymin = 0

            return xmin, ymin

    def __getitem__(self, idx): #idx is filename
        sample = self.patches[idx]
        (x,y), slide, polygons = sample["coords"], sample["slide"], sample["polygons"]

        image = self.get_patch(x,y, slide)
        mask = self.get_y_patch(x,y, slide, polygons)

        # one-hot-encode the mask
        mask = one_hot_encode(mask, self.class_rgb_values)
        
        #apply staining 
        if self.staining: 
            image = get_stained_patch(image,patch_size = self.patch_size, num_patches = self.num_patches)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            #toTensor
            image = image.transpose(2, 0, 1).astype('float32')
            mask = mask.transpose(2, 0, 1).astype('int')
        
        return [image, mask]


    def __len__(self):
        # return length of 
        return len(self.image_paths)
