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
    btn_up = Gtk.Template.Child('up')
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
        self.lsv_spots.connect('activate', self.on_spot_activated)

    def bind_owner(self, tool):
        self.stocraft : Stocraft = tool

    @Gtk.Template.Callback()
    def on_days_value_changed(self,sender):
        # self.stocraft.make_box(int(self.spin_days.get_value()))
        pass

    @Gtk.Template.Callback()
    def on_code_activate(self,sender):
        code = self.entry_code.get_text().strip()
        if code: self.stocraft.cmd_measure(self.entry_code.get_text(),int(self.spin_days.get_value()),func=self.measure)

    @Gtk.Template.Callback()
    def on_score_value_changed(self,sender):
        for i in range(self.model.get_n_items()):
            self.model.remove(0)

        for row in self.rows[1:]:
            if float(row[5]) >= self.spin_score.get_value():
                self.model.append(','.join(row))

    @Gtk.Template.Callback()
    def on_up_toggled(self,sender):
        if sender.get_active():
            sender.set_label('停止')
            self.stocraft.cmd_up(int(self.spin_days.get_value()),func=self.update_stocks)
        else:
            sender.set_label('选股')
            self.stocraft.cmd_up_stop()

    def setup_listitem(self, factory, lsi):
        lsi.set_child(Gtk.Label())

    def bind_listitem(self, factory, lsi):
        label = lsi.get_child()
        label.set_halign(Gtk.Align.START)
        line = lsi.get_item().get_string()
        编号,日期,天数,代码,名字,评分,资金分布,套牢盘,套牢金,支撑价,底部评分 = *line.split(','),

        label.set_text(f'{代码} {名字}\n套牢盘：{套牢盘} {套牢金}\n支撑价：{支撑价}\n资金分布：{资金分布} 评分：{评分}')

    def on_spot_activated(self, lsv, index):
        item = self.selection.get_selected_item()
        if item:
            row = item.get_string().split(',')
            self.entry_code.set_text(row[3])
            self.entry_code.emit('activate')

    def update_stocks(self,line):
        row = line.split(',')
        if self.rows: 
            if float(row[5]) > self.spin_score.get_value(): GLib.idle_add(self.model.append,line)

        self.rows.append(row)

    def measure(self,d,f):
        # self.features.append(row)
        pass