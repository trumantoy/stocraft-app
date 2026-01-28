from typing import Literal, TypedDict
import pygfx as gfx
# import pybullet as bullet

import numpy as np
import pylinalg as la
from ..tools.builtin import *

class Editor(gfx.Scene):
    def __init__(self):
        super().__init__()
        self.steps = list()

        self.toolbar = list()
        self.actionbar = list()

        ortho_camera = gfx.OrthographicCamera()
        persp_camera = gfx.PerspectiveCamera()
        persp_camera.local.position = ortho_camera.local.position = [0,-0.8,0.2]
        ortho_camera.show_pos([0,0,0],up=[0,0,1])
        persp_camera.show_pos([0,0,0],up=[0,0,1])
        self.persp_camera = persp_camera
        self.ortho_camera = ortho_camera
        
        self.camera_controller = gfx.OrbitController()
        self.camera_controller.add_camera(self.persp_camera)
        self.camera_controller.add_camera(self.ortho_camera)

        self.skybox = SkyBox()
        self.add(self.skybox)

        self.ground = Ground()
        self.ground.local.z -= 0.001
        self.add(self.ground)

        ambient = gfx.AmbientLight()
        self.add(ambient)

        light = gfx.DirectionalLight()
        light.local.position = (0.2, -1, 0.3)
        light.shadow.camera.width = light.shadow.camera.height = 1
        self.add(light)

    def step(self,dt=1/180,*args):
        if self.steps:
            self.steps[0]()
            self.steps.pop(0)

        for entity in self.children:
            if 'step' not in dir(entity): continue 
            entity.step(dt,*args)
