import gi

from simtoy.tools.stocraft import Stocraft
gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk, GObject, Gio, Gdk

import threading
import pandas as pd
import pygfx as gfx

@Gtk.Template(filename='ui/panel.ui')
class Panel (Gtk.ScrolledWindow):
    __gtype_name__ = "Panel"
    provider = Gtk.CssProvider.new()

    lsv_spots = Gtk.Template.Child('spots')
    btn_candidate = Gtk.Template.Child('candidate')
    entry_code = Gtk.Template.Child('code')
    spin_score = Gtk.Template.Child('score')
    spin_days = Gtk.Template.Child('days')
    
    def __init__(self):
        self.provider.load_from_path('ui/panel.css')
        Gtk.StyleContext.add_provider_for_display(self.get_display(),self.provider,Gtk.STYLE_PROVIDER_PRIORITY_USER)

        self.model = Gtk.StringList()
        self.selection = Gtk.SingleSelection.new(self.model)
        self.selection.set_autoselect(True)
        self.selection.set_can_unselect(True)

        factory = Gtk.SignalListItemFactory()
        factory.connect("setup", self.setup_listitem)
        factory.connect("bind", self.bind_listitem)
        self.lsv_spots.set_factory(factory)
        self.lsv_spots.set_model(self.selection)
        self.rows = []
        self.features = []

    def bind_owner(self, tool):
        self.stocraft : Stocraft = tool

    @Gtk.Template.Callback()
    def on_code_activate(self,sender):
        self.stocraft.cmd_measure(self.entry_code.get_text(),int(self.spin_days.get_value()),func=self.measure)

    @Gtk.Template.Callback()
    def on_score_value_changed(self,sender):
        self.model.remove_all()
        df = pd.DataFrame(self.rows[1:],columns=self.rows[0])
        df = df[df['score'] >= self.spin_score.get_value()]
        for row in df.values:
            GLib.idle_add(self.model.append,','.join(row))

    @Gtk.Template.Callback()
    def on_btn_candidate_clicked(self,sender):
        self.stocraft.cmd_up(int(self.spin_days.get_value()),func=self.update_stocks)

    def setup_listitem(self, factory, lsi):
        lsi.set_child(Gtk.Label())

    def bind_listitem(self, factory, lsi):
        label = lsi.get_child()
        label.set_halign(Gtk.Align.START)
        row = lsi.get_item().get_string()
        label.set_text(row)

    def update_stocks(self,line):
        row = line.split(',')
        if self.rows: 
            if float(row[4]) > self.spin_score.get_value(): GLib.idle_add(self.model.append,line)

        self.rows.append(row)

    def measure(self,line):
        row = line.split(',')
        if self.features: 
            # GLib.idle_add(self.model.append,line)
            pass
        self.features.append(row)
