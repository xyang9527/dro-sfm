# Visualization code, vtk point cloud and camera visualization adapted from 
# DeMoN - Depth Motion Network https://github.com/lmb-freiburg/demon

import numpy as np
import time
import vtk
import os
import logging

from multiprocessing import Process, Queue
import matplotlib.pyplot as plt


def gray2rgb(im, cmap='gray'):
    cmap = plt.get_cmap(cmap)
    rgba_img = cmap(im.astype(np.float32))
    rgb_img = np.delete(rgba_img, 3, 2)
    return rgb_img


def normalize_depth_for_display(depth, pc=98, crop_percent=0, normalizer=None, cmap='gray'):
    # convert to disparity
    vinds = depth>0
    depth = 1./(depth + 1)

    z1 = np.percentile(depth[vinds], pc)
    z2 = np.percentile(depth[vinds], 100-pc)

    depth = (depth - z2) / (z1 - z2)
    depth = np.clip(depth, 0, 1)

    depth = gray2rgb(depth, cmap=cmap)
    keep_H = int(depth.shape[0] * (1-crop_percent))
    depth = depth[:keep_H]
    return depth


def create_image_depth_figure(image, depth):
    depth_image = 255 * normalize_depth_for_display(depth)
    figure = np.concatenate([image, depth_image], axis=1)
    return figure


def create_camera_polydata(R, t, only_polys=False):
    """Creates a vtkPolyData object with a camera mesh: https://github.com/lmb-freiburg/demon"""
    import vtk
    cam_points = np.array([ 
        [0, 0, 0],
        [-1,-1, 1.5],
        [ 1,-1, 1.5],
        [ 1, 1, 1.5],
        [-1, 1, 1.5],
        [-0.5, 1, 1.5],
        [ 0.5, 1, 1.5],
        [ 0,1.2,1.5],
        [ 1,-0.5,1.5],
        [ 1, 0.5,1.5],
        [ 1.2, 0, 1.5]]
    ) 
    cam_points = (0.05*cam_points - t).dot(R)

    vpoints = vtk.vtkPoints()
    vpoints.SetNumberOfPoints(cam_points.shape[0])
    for i in range(cam_points.shape[0]):
        vpoints.SetPoint(i, cam_points[i])
    vpoly = vtk.vtkPolyData()
    vpoly.SetPoints(vpoints)
    
    poly_cells = vtk.vtkCellArray()

    if not only_polys:
        line_cells = vtk.vtkCellArray()
        
        line_cells.InsertNextCell( 5 );
        line_cells.InsertCellPoint( 1 );
        line_cells.InsertCellPoint( 2 );
        line_cells.InsertCellPoint( 3 );
        line_cells.InsertCellPoint( 4 );
        line_cells.InsertCellPoint( 1 );

        line_cells.InsertNextCell( 3 );
        line_cells.InsertCellPoint( 1 );
        line_cells.InsertCellPoint( 0 );
        line_cells.InsertCellPoint( 2 );

        line_cells.InsertNextCell( 3 );
        line_cells.InsertCellPoint( 3 );
        line_cells.InsertCellPoint( 0 );
        line_cells.InsertCellPoint( 4 );

        # x-axis indicator
        line_cells.InsertNextCell( 3 );
        line_cells.InsertCellPoint( 8 );
        line_cells.InsertCellPoint( 10 );
        line_cells.InsertCellPoint( 9 );
        vpoly.SetLines(line_cells)
    else:
        # left
        poly_cells.InsertNextCell( 3 );
        poly_cells.InsertCellPoint( 0 );
        poly_cells.InsertCellPoint( 1 );
        poly_cells.InsertCellPoint( 4 );

        # right
        poly_cells.InsertNextCell( 3 );
        poly_cells.InsertCellPoint( 0 );
        poly_cells.InsertCellPoint( 3 );
        poly_cells.InsertCellPoint( 2 );

        # top
        poly_cells.InsertNextCell( 3 );
        poly_cells.InsertCellPoint( 0 );
        poly_cells.InsertCellPoint( 4 );
        poly_cells.InsertCellPoint( 3 );

        # bottom
        poly_cells.InsertNextCell( 3 );
        poly_cells.InsertCellPoint( 0 );
        poly_cells.InsertCellPoint( 2 );
        poly_cells.InsertCellPoint( 1 );

        # x-axis indicator
        poly_cells.InsertNextCell( 3 );
        poly_cells.InsertCellPoint( 8 );
        poly_cells.InsertCellPoint( 10 );
        poly_cells.InsertCellPoint( 9 );

    # up vector (y-axis)
    poly_cells.InsertNextCell( 3 );
    poly_cells.InsertCellPoint( 5 );
    poly_cells.InsertCellPoint( 6 );
    poly_cells.InsertCellPoint( 7 );

    vpoly.SetPolys(poly_cells)

    return vpoly


def create_camera_actor(R, t):
    """https://github.com/lmb-freiburg/demon 
    Creates a vtkActor object with a camera mesh
    """
    vpoly = create_camera_polydata(R, t)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(vpoly)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().LightingOff()
    actor.GetProperty().SetLineWidth(2)

    return actor


def create_pointcloud_polydata(points, colors=None):
    """https://github.com/lmb-freiburg/demon
    Creates a vtkPolyData object with the point cloud from numpy arrays
    
    points: numpy.ndarray
        pointcloud with shape (n,3)
    
    colors: numpy.ndarray
        uint8 array with colors for each point. shape is (n,3)

    Returns vtkPolyData object
    """
    logging.warning(f'create_pointcloud_polydata(points={points.shape}, colors={colors.shape})')
    vpoints = vtk.vtkPoints()
    vpoints.SetNumberOfPoints(points.shape[0])

    for i in range(points.shape[0]):
        vpoints.SetPoint(i, points[i])

    vpoly = vtk.vtkPolyData()
    vpoly.SetPoints(vpoints)
    
    if not colors is None:
        vcolors = vtk.vtkUnsignedCharArray()
        vcolors.SetNumberOfComponents(3)
        vcolors.SetName("Colors")
        vcolors.SetNumberOfTuples(points.shape[0])
        for i in range(points.shape[0]):
            vcolors.SetTuple3(i ,colors[i,0],colors[i,1], colors[i,2])
        vpoly.GetPointData().SetScalars(vcolors)

    vcells = vtk.vtkCellArray()
    
    for i in range(points.shape[0]):
        vcells.InsertNextCell(1)
        vcells.InsertCellPoint(i)
        
    vpoly.SetVerts(vcells)
    
    return vpoly


def create_pointcloud_actor(points, colors=None):
    """Creates a vtkActor with the point cloud from numpy arrays
    
    points: numpy.ndarray
        pointcloud with shape (n,3)
    
    colors: numpy.ndarray
        uint8 array with colors for each point. shape is (n,3)

    Returns vtkActor object
    """
    vpoly = create_pointcloud_polydata(points, colors)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(vpoly)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(5)

    return actor


def visualize_prediction(pointcloud, colors, poses=None, renwin=None):
    """ render point cloud and cameras """

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0, 0, 0)

    pointcloud_actor = create_pointcloud_actor(points=pointcloud, colors=colors)
    pointcloud_actor.GetProperty().SetPointSize(2)
    renderer.AddActor(pointcloud_actor)

    for pose in poses:
        R, t = pose[:3, :3], pose[:3, 3]
        cam_actor = create_camera_actor(R,t)
        cam_actor.GetProperty().SetColor((255, 255, 0))
        renderer.AddActor(cam_actor)

    camera = vtk.vtkCamera()
    camera.SetPosition((1, -1, -2));
    camera.SetViewUp((0, -1, 0));
    camera.SetFocalPoint((0, 0, 2));

    renderer.SetActiveCamera(camera)
    renwin = vtk.vtkRenderWindow()

    renwin.SetWindowName("Point Cloud Viewer")
    renwin.SetSize(800,600)
    renwin.AddRenderer(renderer)
 
    # An interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interstyle = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(interstyle)
    interactor.SetRenderWindow(renwin)

    # Render and interact
    renwin.Render()
    interactor.Initialize()
    interactor.Start()


class vtkTimerCallback():
    def __init__(self, cinematic=False, render_path=None, clear_points=False, is_kitti=False):
        logging.warning(f'vtkTimerCallback::__init__(cinematic={cinematic}, render_path={render_path}, clear_points={clear_points}, is_kitti={is_kitti})')
        self.timer_count = 0
        self.write_count = 0
        self.null_loop_count = 0

        self.cinematic = cinematic
        self.render_path = render_path
        self.clear_points = clear_points
        self.is_kitti = is_kitti

        self.point_actor = None
        self.pos = None
        self.pt = None
        self.alpha = 0.2

    def execute(self, obj, event):
        logging.warning(f'vtkTimerCallback::execute(..)')

        is_null_loop = True
        while not self.queue.empty():
            is_null_loop = False
            renwin = obj.GetRenderWindow()
            renderer = renwin.GetRenderers().GetFirstRenderer()

            pointcloud, pose = self.queue.get(False)

            if pointcloud is not None:
                if (self.point_actor is not None) and self.clear_points:
                    renderer.RemoveActor(self.point_actor)

                points, colors = pointcloud[0], pointcloud[1]
                pointcloud_actor = create_pointcloud_actor(points, colors)

                renderer.AddActor(pointcloud_actor)
                self.point_actor = pointcloud_actor

            if pose is not None:
                R, t = pose[:3, :3], pose[:3, 3]
                cam_actor = create_camera_actor(R,t)
                cam_actor.GetProperty().SetColor((255, 255, 0))
                renderer.AddActor(cam_actor)

                if self.cinematic:
                    camera = renderer.GetActiveCamera()

                    if self.is_kitti:
                        pos = np.array([-0.3, -3.0, -8.0, 1])
                    else:
                        pos = np.array([-0.3, -0.3, -1.0, 1])
                    
                    pos = np.dot(np.linalg.inv(pose)[:3], pos)

                    if self.pos is None:
                        self.pos = pos

                    # self.pos = (1-self.alpha) * self.pos + self.alpha * pos
                    # camera.SetPosition(*self.pos)
                    # pose  = np.linalg.inv(pose)
                    R, T = pose[:3, :3], pose[:3, 3:4]
                    pos_our = -1 * np.matmul(R.transpose(), T)
                    pos_our = pos_our[:, 0]

                    forward = np.matmul(np.array([0, 0, 1]).reshape(1, 3), R)[0]
                    up = np.array([0, -1, 0])

                    if not self.is_kitti:
                        camera.SetPosition(pos_our - 1 * forward + 0.5 *up)                    
                        camera.SetFocalPoint(pos_our + 1 * forward + -0.1 * up)
                    else:
                        # camera.SetPosition(pos_our - 50 * forward + 480 *up)                    
                        # camera.SetFocalPoint(pos_our + 1 * forward + -1 * up)
                        camera.SetPosition(pos_our - 5 * forward + 48 *up)                    
                        camera.SetFocalPoint(pos_our + 1 * forward + -1 * up)

                    pt = np.array([0, 0.5, 3, 1])
                    pt = np.dot(np.linalg.inv(pose)[:3], pt)

                    if self.pt is None:
                        self.pt = pt

                    self.pt = (1-self.alpha) * self.pt + self.alpha * pt
                    # camera.SetFocalPoint(*self.pt)

            renwin.Render()

            if (pose is not None) and (self.render_path is not None):
                w2if = vtk.vtkWindowToImageFilter()
                w2if.SetInput(renwin)
                w2if.Update()

                writer = vtk.vtkPNGWriter()

                output_file = os.path.join(self.render_path, "%06d.png"%self.write_count)
                writer.SetFileName(output_file)
                writer.SetInputData(w2if.GetOutput())
                writer.Write()
                logging.info(f'  write {output_file}')

                self.write_count += 1
                self.null_loop_count = 0

        if is_null_loop:
            self.null_loop_count += 1
        self.timer_count += 1

        logging.info(f'  self.write_count:      {self.write_count}')
        logging.info(f'  self.timer_count:      {self.timer_count}')
        logging.info(f'  self.null_loop_count:  {self.null_loop_count}')

        if self.null_loop_count > 2 and self.write_count > 1:
            # https://stackoverflow.com/questions/15639762/close-vtk-window-python
            # close window
            logging.warning(f'  close vtk window')
            renwin = obj.GetRenderWindow()
            renwin.Finalize()
            logging.info(f'  renwin.Finalize() done.')
            obj.TerminateApp()
            logging.info(f'  obj.TerminateApp() done.')
            # del renwin, obj


class InteractiveViz(Process):
    def __init__(self, queue, cinematic, render_path, clear_points, win_size, is_kitti=False):
        logging.warning(f'InteractiveViz::__init__(queue={queue.qsize()}, cinematic={cinematic}, render_path={render_path},'
                        f'\n\tclear_points={clear_points}, win_size={win_size}, is_kitti={is_kitti})')
        super(InteractiveViz, self).__init__()
        self.queue = queue
        self.cinematic = cinematic
        self.render_path = render_path
        self.clear_points = clear_points
        self.is_kitti = is_kitti
        self.win_size = win_size
        logging.info(f'  win_size: {win_size}')

    def run(self):
        logging.warning(f'InteractiveViz::run()')
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(0, 0, 0)

        if self.is_kitti:
            camera = vtk.vtkCamera()
            camera.SetPosition((0, 0, -2));
            camera.SetViewUp((0, -1, 0));
            camera.SetFocalPoint((0, 0, 2));
            renderer.SetActiveCamera(camera)
        else:
            camera = vtk.vtkCamera()
            camera.SetPosition((1, -1, -2));
            camera.SetViewUp((0, -1, 0));
            camera.SetFocalPoint((0, 0, 2));
            renderer.SetActiveCamera(camera)

        renwin = vtk.vtkRenderWindow()
        renwin.SetWindowName("Point Cloud Viewer")

        # renwin.SetSize(800, 600)
        # renwin.SetSize(640, 480)  # matterport
        # renwin.SetSize(1296, 968)  # scannet
        win_h, win_w = self.win_size
        renwin.SetSize(win_w, win_h)

        renwin.AddRenderer(renderer)

        interactor = vtk.vtkRenderWindowInteractor()
        interstyle = vtk.vtkInteractorStyleTrackballCamera()
        interactor.SetInteractorStyle(interstyle)
        interactor.SetRenderWindow(renwin)

        interactor.Initialize()

        cb = vtkTimerCallback(self.cinematic, self.render_path, self.clear_points, self.is_kitti)
        cb.queue = self.queue

        interactor.AddObserver('TimerEvent', cb.execute)
        timerId = interactor.CreateRepeatingTimer(100);

        logging.info(f'  renderer:   {type(renderer)}')
        logging.info(f'  renwin:     {type(renwin)}')
        logging.info(f'  interactor: {type(interactor)}')

        #start the interaction and timer
        interactor.Start()
