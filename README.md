run tests:
    pytest --pdb --cov=. --cov-report html:htmlcov ./detutils/test



yolo scales:
    coord_scale: float = 5.0,
    obj_scale: float = 1.0,
    noobj_scale: float = 0.5,
    class_scale: float = 1.0,
    objective_conf_thre: float = 0.5

Develop YOLOv4：MAKE SURE THAT IMAGES ARE NOT TRANSPOSED AFTER DATA LOADING, SHOULD BE ALIGNED WITH BOX