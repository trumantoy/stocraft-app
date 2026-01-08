import gi
gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk, Gio

import cairo
import numpy as np
import pygfx as gfx
from rendercanvas.offscreen import RenderCanvas
from pathlib import Path

from panel import *

@Gtk.Template(filename='ui/app_window.ui')
class AppWindow (Gtk.ApplicationWindow):
    __gtype_name__ = "AppWindow"

    paned : Gtk.Paned = Gtk.Template.Child('paned')
    stack : Gtk.Stack = Gtk.Template.Child('panel')
    area : Gtk.DrawingArea = Gtk.Template.Child('widget')

    def __init__(self):
        provider = Gtk.CssProvider.new()
        Gtk.StyleContext.add_provider_for_display(self.get_display(),provider,Gtk.STYLE_PROVIDER_PRIORITY_USER)

        # self.scene = self.editor = Editor()
        # self.canvas = RenderCanvas(size=(1024,768))
        # self.renderer = gfx.renderers.WgpuRenderer(self.canvas)
        
        # self.panel : Panel = self.stack.get_visible_child()
        # self.area.set_draw_func(self.draw)

        # zoom_controller = Gtk.EventControllerScroll.new(Gtk.EventControllerScrollFlags(Gtk.EventControllerScrollFlags.VERTICAL))
        # zoom_controller.connect("scroll", lambda sender,dx,dy: self.renderer.convert_event(dict(event_type='wheel',dx=0.0,dy=dy*100,x=0,y=0,time_stamp=time.perf_counter())))
        
        # click_controller = Gtk.GestureClick.new()
        # click_controller.set_button(1)
        # click_controller.connect("pressed", lambda sender,n_press,x,y: self.renderer.convert_event(dict(event_type='pointer_down',x=x ,y=y,button=3,buttons=(3,),time_stamp=time.perf_counter())))
        # click_controller.connect("released", lambda sender,n_press,x,y: self.renderer.convert_event(dict(event_type='pointer_up',x=x ,y=y,button=3,buttons=(3,),time_stamp=time.perf_counter())))

        # rotation_controller = Gtk.GestureClick.new()
        # rotation_controller.set_button(2)
        # rotation_controller.connect("pressed", lambda sender,n_press,x,y: self.renderer.convert_event(dict(event_type='pointer_down',x=x ,y=y,button=1,buttons=(1,),time_stamp=time.perf_counter())))
        # rotation_controller.connect("released", lambda sender,n_press,x,y: self.renderer.convert_event(dict(event_type='pointer_up',x=x ,y=y,button=1,buttons=(1,),time_stamp=time.perf_counter())))

        # pan_controller = Gtk.GestureClick.new()
        # pan_controller.set_button(3)
        # pan_controller.connect("pressed", lambda sender,n_press,x,y: self.renderer.convert_event(dict(event_type='pointer_down',x=x,y=y,button=2,buttons=(2,),time_stamp=time.perf_counter())))
        # pan_controller.connect("released", lambda sender,n_press,x,y: self.renderer.convert_event(dict(event_type='pointer_up',x=x,y=y,button=2,buttons=(2,),time_stamp=time.perf_counter())))

        # motion_controller = Gtk.EventControllerMotion()
        # motion_controller.connect("motion", lambda sender,x,y: self.renderer.convert_event(dict(event_type='pointer_move',x=x ,y=y,time_stamp=time.perf_counter())))

        # if click_controller: self.area.add_controller(click_controller)
        # if rotation_controller: self.area.add_controller(rotation_controller)
        # if pan_controller: self.area.add_controller(pan_controller)
        # if zoom_controller: self.area.add_controller(zoom_controller)
        # if motion_controller: self.area.add_controller(motion_controller)

        # action = Gio.SimpleAction.new('import', None)
        # action.connect('activate', self.file_import)
        # self.add_action(action)

        # action = Gio.SimpleAction.new('export', None)
        # action.connect('activate', self.file_export)
        # self.add_action(action)

        # action = Gio.SimpleAction.new('close', None)
        # action.connect('activate', self.file_close)
        # self.add_action(action)

        # self.light = gfx.PointLight(intensity=1)
        # self.editor.add(self.light)

        # self.tool = Engravtor(name='M3-00-355')
        # self.tool.set_consumable('木板-100x100x1')
        # self.editor.add(self.tool)
        
        # self.camera_controller = gfx.OrbitController()
        # for c in self.tool.get_viewport(): self.camera_controller.add_camera(c)
        # self.hotbar.connect('item-added', self.item_added)
        # self.hotbar.bind_owner(self.tool)
        # self.panel.bind_owner(self.tool)

        # self.tool.selected_func = self.selected_func
        # self.tool.transformed_func = self.transformed_func
        # self.propbar.connect('item-removed', self.item_removed)
        # # self.panel.add_device(self.tool)
        # self.panel.connect('preview', self.preview)

        # GLib.timeout_add(1000/180,lambda: self.editor.step(1/60) or True)

    def do_size_allocate(self, width: int, height: int, baseline: int):
        if hasattr(self,'prev_width'): 
            panel = self.stack.get_visible_child()
            prev_panel_width = self.prev_width - self.paned.get_position()
            self.paned.set_position(width - prev_panel_width)

        self.prev_width = width
        Gtk.ApplicationWindow.do_size_allocate(self,width,height,baseline)

    def draw(self,area, cr : cairo.Context, area_w, area_h):
        width,height = self.canvas.get_physical_size()

        if width != area_w or height != area_h: 
            self.canvas = RenderCanvas(size=(area_w,area_h))
            self.renderer = gfx.renderers.WgpuRenderer(self.canvas)
            self.camera_controller.register_events(self.renderer)
        
        camera = self.camera_controller.cameras[0]
        camera.logical_size = (area_w,area_h)
        self.light.local.position = camera.local.position
        self.renderer.render(self.editor, camera)
        
        img : np.ndarray = np.asarray(self.canvas.draw())
        img_h,img_w,img_ch = img.shape
        img = np.asarray(img[..., [2, 1, 0, 3]]).copy()
        
        stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_ARGB32, img_w)
        surface = cairo.ImageSurface.create_for_data(img.data, cairo.FORMAT_ARGB32, img_w, img_h, stride)
        cr.set_source_surface(surface, 0, 0)

        cr.paint()

        GLib.idle_add(area.queue_draw)
