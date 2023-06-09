U
    ��oc�I  �                   @   s6   d dl Z d dlZd dlT d dlmZ G dd� d�ZdS )�    N)�*)�geometryc                   @   s�   e Zd Zdeeeed�dd�Zedd� �Zejd	d� �Zed
d� �Z	edd� �Z
dd� Zdeed�dd�Zdeed�dd�Zdd� ZdS )�SlideContainerr   �   N)�file�level�width�heightc	              
      s  d�_ d�_d�_� �_t|��v}	t�|	�}
tt�fdd�|
d D ��fdd�|
d D ����_	� fdd�|
d	 D �d
 ��fdd�|
d D ��_
W 5 Q R X t�fdd��j
D ���_�jdk�r
�j��j	d � �j��j	d � �j��j	d � �j��j	d � t�t�j���_t�t�j�dtt�j�� ��_t�t� ���_t�t��j�d�jjd �jjd ��d d �d d �d d�f tj�}t�|dd
�}t� |d
dtj!tj" �\�_#}|�_$|�_%t&d�j%d�j$� �jj'| �_(|d k�r�jjd }|�_)|�_*|�_+|�_,d S )NZtumor_idZid_superZsupercategoryc                    s   g | ]}|� j  �qS � ��name��.0�cat��selfr
   �O/home/klose/CanineCutaneousTumors/segmentation/../slide/full_slide_container.py�
<listcomp>!   s     z+SlideContainer.__init__.<locals>.<listcomp>�
categoriesc                    s   g | ]}|� j  �qS r
   )�idr   r   r
   r   r   !   s    c                    s"   g | ]}|d  � j kr|d �qS )�	file_namer   r   )r   �i)r   r
   r   r   &   s    ��imagesr   c                    s   g | ]}|d  � kr|�qS ��image_idr
   )r   �annor   r
   r   r   *   s    ��annotationsc                    s   g | ]}|� j  �qS r
   ��poly_klasse�r   �polyr   r
   r   r   1   s     ZAnnotationskasten�IgnorezKasten - vereinfachtzKasten - Test�   �r   r   ������   )�   r&   ��   z
self heigtz
self width)-r   r   r   r   �open�json�load�dict�zip�tissue_classes�polygons�set�labels�discard�fromkeys�listZtraining_dict�len�probabilities�	openslideZ
open_slide�str�slide�cv2�cvtColor�np�array�read_regionZlevel_count�level_dimensions�COLOR_RGB2GRAY�GaussianBlur�	thresholdZTHRESH_BINARYZTHRESH_OTSU�whiter   r	   �print�level_downsamples�down_factor�_level�sample_func�dataset_type�
label_dict)r   r   Zannotation_filer   r   r	   rG   rH   rI   �f�data�	thumbnailZblurred�_r
   )r   r   r   r   �__init__
   sd    

	"�
��" �2��   
�

zSlideContainer.__init__c                 C   s   | j S �N�rF   r   r
   r
   r   r   n   s    zSlideContainer.levelc                 C   s   | j j| | _|| _d S rO   )r8   rD   rE   rF   )r   �valuer
   r
   r   r   r   s    c                 C   s   | j | jfS rO   )r   r	   r   r
   r
   r   �shapew   s    zSlideContainer.shapec                 C   s   | j j| j S rO   )r8   r>   rF   r   r
   r
   r   �slide_shape|   s    zSlideContainer.slide_shapec                 C   s   | j S rO   rP   r   r
   r
   r   �get_new_level�   s    zSlideContainer.get_new_level)�x�yc                 C   sH   t �| jjt|�t|�f| j| j| jfd��d d �d d �d d�f }|S )N)�locationr   �sizer%   )r;   r<   r8   r=   �intrF   r   r	   )r   rU   rV   �rgbr
   r
   r   �	get_patch�   s     
��zSlideContainer.get_patchc           
      C   s�   dt j| j| jft jd� }dd� | j�� D �}| jD ]^}t �|d ��	d�| j
 }|||f }| j||| j   }t�||�	d��t�gd|d� q6| jdkr�t�| �||�tj�| jk}|dk}	d|t �||	�< |S )	Nr$   )rR   �dtypec                 S   s   i | ]\}}||�qS r
   r
   �r   �k�vr
   r
   r   �
<dictcomp>�   s      z.SlideContainer.get_y_patch.<locals>.<dictcomp>�segmentation�r$   �   )r$   r"   rc   r   )r;   �onesr	   r   �int8r-   �itemsr.   r<   �reshaperE   rI   r   r9   ZdrawContours�astyperY   rH   r:   r[   r?   rB   �logical_and)
r   rU   rV   Zy_patch�inv_mapr    �coordinates�labelZ
white_mask�excludedr
   r
   r   �get_y_patch�   s>    
��
��  �
 ���zSlideContainer.get_y_patchc                    s�  dd� �j �� D �}t�j�rB�j�jf�j�j�jj�j	d��S d\}}d}|�s�d}t
�t�j�� �t�j�� ��d � � �fdd��jD �}d	d� |D �}t�|�t|� }t
�||�d }t�|d
 ��d�}	|d \}
}}}|dk rN|sNd}d}d}�jdkr�|r�t���||��t�tj��||�dd�d � �j|�   k�s~t���||��j|�   k��j�j d k r�d}q�qN||fS )Nc                 S   s   i | ]\}}||�qS r
   r
   r]   r
   r
   r   r`   �   s      z<SlideContainer.get_new_train_coordinates.<locals>.<dictcomp>)�classesrX   r>   r   r#   Tr   c                    s   g | ]}|�j  � kr|�qS r
   r   r   �rl   r   r
   r   r   �   s     z<SlideContainer.get_new_train_coordinates.<locals>.<listcomp>c                 S   s   g | ]}|d  �qS )�arear
   r   r
   r
   r   r   �   s     ra   rb   �bbox�   �classification)�return_countsr"   g�������?F)r-   rf   �callablerG   r.   r0   rR   r8   r>   r   �random�choicesr3   r5   �keys�valuesr;   r<   �sumrg   rH   �uniquern   �argmaxrI   r   r	   )r   rj   �xmin�ymin�found�iterr.   Zpolygons_area�polygonrk   �minx�miny�xrange�yranger
   rp   r   �get_new_train_coordinates�   sR    
���������z(SlideContainer.get_new_train_coordinates)r   r   r   NNN)r   r   )r   r   )�__name__�
__module__�__qualname__�PathrY   rN   �propertyr   �setterrR   rS   rT   r[   rn   r�   r
   r
   r
   r   r      s.            � �d



r   )r6   r9   �fastai.visionZshapelyr   r   r
   r
   r
   r   �<module>   s   