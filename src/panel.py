import gi

from simtoy.tools.stocraft import Stocraft
gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk, GObject, Gio, Gdk

import threading
import pygfx as gfx

@Gtk.Template(filename='ui/panel.ui')
class Panel (Gtk.ScrolledWindow):
    __gtype_name__ = "Panel"
    provider = Gtk.CssProvider.new()

    lsv_spots = Gtk.Template.Child('spots')
    btn_candidate = Gtk.Template.Child('candidate')
    entry_code = Gtk.Template.Child('code')
    
    def __init__(self):
        self.provider.load_from_path('ui/panel.css')
        Gtk.StyleContext.add_provider_for_display(self.get_display(),self.provider,Gtk.STYLE_PROVIDER_PRIORITY_USER)

    def bind_owner(self, tool):
        self.stocraft : Stocraft = tool

    @Gtk.Template.Callback()
    def on_btn_candidate_clicked(self,sender):
        self.stocraft.cmd_up()

        self.df = self.stocraft.stocks
