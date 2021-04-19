run tests:
    pytest --pdb --cov=. --cov-report html:htmlcov <path_to_test_script>


yolov4 scales:
    coord_scale: float = 5.0,
    obj_scale: float = 1.0,
    noobj_scale: float = 0.5,
    class_scale: float = 1.0,
    objective_conf_thre: float = 0.5
