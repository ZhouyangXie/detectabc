installation:
    for downloaded detectabc.zip:
        > pip install detectabc.zip

requirements:
    numpy
    torch>=1.5.0
    torchvision>=0.6.0

packages:
    detectabc/detutils: storage, transforms and metrics of bounding boxes.
    detectabc/modules: useful PyTorch modules or models to build a detector.
    detectabc/modules/backbone: backbones for detector.
    detectabc/modules/neck: feature aggreagation modules such as FPN, PAN etc.
    detectabc/modules/head: functionals to convert dense prections to box predictions.
    detectabc/modules/functional: functionals as a part of backbone or neck.
    detectabc/modules/detector: off-the-shelf detectors.
    detectabc/modules/contrib: 3rd-party module implementations.

run tests:
    pytest --pdb --cov=. --cov-report html:htmlcov <path_to_test_script>
