# Segmentation_ROS_View
  File "/home/parksungjun/torchvision/setup.py", line 10, in <module>
    import torch
  File "/home/parksungjun/.local/lib/python3.8/site-packages/torch/__init__.py", line 216, in <module>
    raise ImportError(textwrap.dedent('''
ImportError: Failed to load PyTorch C extensions:
    It appears that PyTorch has loaded the `torch/_C` folder
    of the PyTorch repository rather than the C extensions which
    are expected in the `torch._C` namespace. This can occur when
    using the `install` workflow. e.g.
        $ python setup.py install && python -c "import torch"

    This error can generally be solved using the `develop` workflow
        $ python setup.py develop && python -c "import torch"  # This should succeed
    or by running Python from a different directory.
what error to talk korean
