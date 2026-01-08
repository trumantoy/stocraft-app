import gi
gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk, GObject, Gio, Gdk

import threading
import pygfx as gfx

@Gtk.Template(filename='ui/panel.ui')
class Panel (Gtk.ScrolledWindow):
    __gtype_name__ = "Panel"
    provider = Gtk.CssProvider.new()
    
    def __init__(self):
        self.provider.load_from_path('ui/panel.css')
        Gtk.StyleContext.add_provider_for_display(self.get_display(),self.provider,Gtk.STYLE_PROVIDER_PRIORITY_USER)

    