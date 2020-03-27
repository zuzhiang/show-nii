import SimpleITK as sitk
import numpy as np


def Get2DTransform(image, rotate_angle=0, scale_width=1, scale_height=1, shear=0, shift_left=0, shift_top=0):
    input_shape = image.GetSize()
    input_shape = np.asarray(input_shape)
    affine_transform = sitk.AffineTransform(2)
    affine_transform.SetCenter(image.TransformIndexToPhysicalPoint([index // 2 for index in image.GetSize()]))
    affine_transform.Rotate(0, 1, 3.1415926 * rotate_angle / 180)
    affine_transform.Scale([2 - scale_width, 2 - scale_height])
    affine_transform.Shear(1, 0, shear)
    affine_transform.Translate([shift_left, shift_top])
    return affine_transform


def Get3DTransform(image, rotate_angle_xy=0, rotate_angle_zx=0, rotate_angle_yz=0,
                   scale_x=1, scale_y=1, scale_z=1,
                   shear=0,
                   shift_x=0, shift_y=0, shift_z=0):
    input_shape = image.GetSize()
    input_shape = np.asarray(input_shape)
    affine_transform = sitk.AffineTransform(3)
    center_point = image.TransformIndexToPhysicalPoint([index // 2 for index in image.GetSize()])
    affine_transform.SetCenter(center_point)

    affine_transform.Rotate(0, 1, 3.1415926 * rotate_angle_xy / 180)
    affine_transform.Rotate(1, 2, 3.1415926 * rotate_angle_yz / 180)
    affine_transform.Rotate(2, 0, 3.1415926 * rotate_angle_zx / 180)
    affine_transform.Scale([1/scale_x, 1/scale_y, 1/scale_z])
    affine_transform.Shear(1, 0, shear)
    affine_transform.Translate([shift_x, shift_y, shift_z])
    return affine_transform


def ApplyTransform(data, transform, size=None, interpolate_method=sitk.sitkBSpline):
    if isinstance(data, np.ndarray):
        image_data = sitk.GetImageFromArray(data)
    elif isinstance(data, sitk.Image):
        image_data = data

    if not (isinstance(size, list) or isinstance(size, list)):
        size = image_data.GetSize()

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image_data)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(interpolate_method)
    print(resampler.GetOutputSpacing())
    # resampler.SetOutputSpacing([index * 2 for index in data.GetSpacing()])

    temp = resampler.Execute(image_data)
    print(temp.GetSpacing())
    print(temp.GetSize())
    result = temp

    # shift = [128, -128, 0]
    # shift_transform = Get3DTransform(temp, shift_x=shift[0], shift_y=shift[1], shift_z=shift[2])
    # print(shift_transform)
    # resampler.SetReferenceImage(temp)
    # resampler.SetSize(size)
    # resampler.SetTransform(shift_transform)
    # result = resampler.Execute(temp)
    # result = sitk.GetArrayFromImage(result)
    return result


def TransformBasedRegistration(fix_image, moving_image, target_image,
                               register_method=sitk.sitkBSpline,
                               transform_method=sitk.sitkNearestNeighbor):
    '''Calculate the transform based on the registration from the moving image
    to the fixed image, then applied this transform on the target image.

    Yang Song, Sep-21-2017'''

    fix_image = sitk.GetImageFromArray(fix_image)
    moving_image = sitk.GetImageFromArray(moving_image)
    target_image = sitk.GetImageFromArray(target_image)

    fix_image = sitk.Normalize(fix_image)
    moving_image = sitk.Normalize(moving_image)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()
    R.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0,
                                               minStep=1e-4,
                                               numberOfIterations=200,
                                               gradientMagnitudeTolerance=1e-8)
    R.SetOptimizerScalesFromIndexShift()
    R.SetInitialTransform(sitk.TranslationTransform(fix_image.GetDimension()))
    R.SetInterpolator(register_method)
    # R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    transform = R.Execute(fix_image, moving_image)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fix_image)
    resampler.SetInterpolator(register_method)
    resampler.SetTransform(transform)
    registered_moving = resampler.Execute(moving_image)
    registered_moving = sitk.GetArrayFromImage(registered_moving)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(target_image)
    resampler.SetInterpolator(transform_method)
    resampler.SetTransform(transform)
    registered_target = resampler.Execute(target_image)
    registered_target = sitk.GetArrayFromImage(registered_target)

    return registered_moving, registered_target, transform