#: Various Python utilities
import os
import sys
import time
import json
import pathlib
from typing import Dict, List, Sequence, Any, Union, Tuple
from collections import OrderedDict

#: Plotting and math modules
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
import matplotlib.colors as colors
import numpy as np
from scipy.linalg import lstsq
from scipy.interpolate import Rbf
from scipy import io
from IPython.display import clear_output

#: Qcodes for running measurements and saving data
import qcodes as qc
from qcodes.station import Station
from qcodes.instrument_drivers.stanford_research.SR830 import SR830
from qcodes.data.io import DiskIO
from datetime import datetime


#: NI DAQ library
import nidaqmx
from nidaqmx.constants import AcquisitionType

#: scanning-squid modules
import squids
import instruments.atto as atto
import utils
from scanner import Scanner
from instruments.daq import DAQAnalogInputs
from plots import ScanPlot, TDCPlot, MSvTPlot
from plots import gatePlot, DoubleGatedPlot
from instruments.lakeshore import Model_372, Model_331, Model_340
from instruments.keithley import Keithley_2400
from instruments.heater import EL320P

from typing import Optional

#: Pint for manipulating physical units
from pint import UnitRegistry
ureg = UnitRegistry()
#: Tell UnitRegistry instance what a Phi0 is, and that Ohm = ohm
with open('squid_units.txt', 'w') as f:
    f.write('Phi0 = 2.067833831e-15 * Wb\n')
    f.write('Ohm = ohm\n')
ureg.load_definitions('./squid_units.txt')

import logging
log = logging.getLogger(__name__)


class Microscope(Station):
    """Base class for scanning SQUID microscope.
    """
    def __init__(self, config_file: str, temp: str, ureg: Any=ureg, log_level: Any=logging.INFO,
                 log_name: str=None, **kwargs) -> None:
        """
        Args:
            config_file: Path to microscope configuration JSON file.
            temp: 'LT' or 'RT', depending on whether the microscope is cold or not.
                Sets the voltage limits for the scanner and Attocubes.
            ureg: pint UnitRegistry for managing physical units.
            log_level: e.g. logging.DEBUG or logging.INFO
            log_name: Log file will be saved as logs/{log_name}.log.
                Default is the name of the microscope configuration file.
            **kwargs: Keyword arguments to be passed to Station constructor.
        """
        super().__init__(**kwargs)
        qc.Instrument.close_all()
        self.config = utils.load_json_ordered(config_file)
        if not os.path.exists('logs'):
            os.mkdir('logs')
        if log_name is None:
            log_file = utils.next_file_name('./logs/' + config_file.split('.')[0], 'log')
        else:
            log_file = utils.next_file_name('./logs/' + log_name, 'log')
        logging.basicConfig(
        	level=logging.INFO,
        	format='%(levelname)s:%(asctime)s:%(module)s:%(message)s',
            datefmt=self.config['info']['timestamp_format'],
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ])
        log.info('Logging started.')
        log.info('Initializing microscope object using file {}.'.format(config_file))

        self.ureg = ureg
        # Callable for converting a string into a quantity with units
        self.Q_ = ureg.Quantity
        self.temp = temp

        #self._add_atto()
        #self._add_ls372()
        #self._add_ls331()
        #self._add_keithley()
        self._add_ke2400_old()
        #self._add_ke2400_b()
        #self._add_ke2400_new()
        self._add_ke2410_b()
        #self._add_ls340()
        self._add_scanner()
        self._add_SQUID()
        self._add_lockins()

    def _add_atto(self):
        """Add Attocube controller to microscope.
        """
        atto_config = self.config['instruments']['atto']
        ts_fmt = self.config['info']['timestamp_format']
        if hasattr(self, 'atto'):
        #     self.atto.clear_instances()
            self.atto.close()
        self.remove_component(atto_config['name'])
        self.atto = atto.ANC300(atto_config, self.temp, self.ureg, ts_fmt)
        self.add_component(self.atto)
        log.info('Attocube controller successfully added to microscope.')

    def _add_ls372(self):
        """Add Lakeshore 372 temperature controller to microscope.
        """
        ls_config = self.config['instruments']['ls372']
        if hasattr(self, 'ls372'):
        #     self.atto.clear_instances()
            self.ls372.close()
        self.remove_component(ls_config['name'])
        self.ls372 = Model_372(ls_config['name'], ls_config['address'])
        self.add_component(self.ls372)
        Tmin_K = ls_config['Tmin_K']
        Tmax_K = ls_config['Tmax_K']
        self.ls372.configure_analog_output('A', Tmin_K, Tmax_K)
        log.info('Lakeshore 372 successfully added to microscope.')

    def _add_ls331(self):
        """Add Lakeshore 331 temperature controller to microscope.
        """
        ls_config = self.config['instruments']['ls331']
        if hasattr(self, 'ls331'):
        #     self.atto.clear_instances()
            self.ls331.close()
        self.remove_component(ls_config['name'])
        self.ls331 = Model_331(ls_config['name'], ls_config['address'])
        self.add_component(self.ls331)
        log.info('Lakeshore 331 successfully added to microscope.')

    def _add_ls340(self):
        """Add Lakeshore 340 temperature controller to microscope.
        """
        ls_config = self.config['instruments']['ls340']
        if hasattr(self, 'ls340'):
        #     self.atto.clear_instances()
            self.ls340.close()
        self.remove_component(ls_config['name'])
        self.ls340 = Model_340(ls_config['name'], ls_config['address'])
        self.add_component(self.ls340)
        log.info('Lakeshore 340 successfully added to microscope.')

    def _add_ke2400_t(self):
        """Add Keithley 2400 SourceMeter to microscope.
        """
        ke_config = self.config['instruments']['ke2400_t']
        if hasattr(self, 'ke2400_t'):
        #     self.ke2400.clear_instances()
            self.ke2400_t.close()
        self.remove_component(ke_config['name'])
        self.ke2400_t = Keithley_2400(ke_config['name'], ke_config['address'])
        self.add_component(self.ke2400_t)
        log.info('top Keithley 2400 successfully added to microscope.')

    def _add_ke2400_old(self):
        """Add Keithley 2400 SourceMeter to microscope.
        """
        ke_config = self.config['instruments']['ke2400_old']
        if hasattr(self, 'ke2400_old'):
        #     self.ke2400.clear_instances()
            self.ke2400_old.close()
        self.remove_component(ke_config['name'])
        self.ke2400_old = Keithley_2400(ke_config['name'], ke_config['address'])
        self.add_component(self.ke2400_old)
        log.info('top Keithley 2400 old successfully added to microscope.')

    def _add_ke2400_new(self):
        """Add Keithley 2400 SourceMeter to microscope.
        """
        ke_config = self.config['instruments']['ke2400_new']
        if hasattr(self, 'ke2400_new'):
        #     self.ke2400.clear_instances()
            self.ke2400_new.close()
        self.remove_component(ke_config['name'])
        self.ke2400_new = Keithley_2400(ke_config['name'], ke_config['address'])
        self.add_component(self.ke2400_new)
        log.info('top Keithley 2400 new successfully added to microscope.')

    def _add_ke2400_b(self):
        """Add second Keithley 2400 SourceMeter to microscope.
        """
        ke_config = self.config['instruments']['ke2400_b']
        if hasattr(self, 'ke2400_b'):
        #     self.ke2400_2.clear_instances()
            self.ke2400_b.close()
        self.remove_component(ke_config['name'])
        self.ke2400_b = Keithley_2400(ke_config['name'], ke_config['address'])
        self.add_component(self.ke2400_b)
        log.info('bottom Keithley 2400_b successfully added to microscope.')

    def _add_ke2410_t(self):
        """Add Keithley 2410 SourceMeter to microscope.
        """
        ke_config = self.config['instruments']['ke2410_t']
        if hasattr(self, 'ke2410_t'):
        #     self.ke2410.clear_instances()
            self.ke2410_t.close()
        self.remove_component(ke_config['name'])
        self.ke2410_t = Keithley_2400(ke_config['name'], ke_config['address'])
        self.add_component(self.ke2410_t)
        log.info('top Keithley 2410_t successfully added to microscope.')

    def _add_ke2410_b(self):
        """Add Keithley 2410 SourceMeter to microscope.
        """
        ke_config = self.config['instruments']['ke2410_b']
        if hasattr(self, 'ke2410_b'):
        #     self.ke2410.clear_instances()
            self.ke2410_b.close()
        self.remove_component(ke_config['name'])
        self.ke2410_b = Keithley_2400(ke_config['name'], ke_config['address'])
        self.add_component(self.ke2410_b)
        log.info('bottom Keithley 2410_b successfully added to microscope.')

    def _add_scanner(self):
        """Add scanner instrument to microscope.
        """
        scanner_config = self.config['instruments']['scanner']
        daq_config = self.config['instruments']['daq']
        if hasattr(self, 'scanner'):
            #self.scanner.clear_instances()
            self.scanner.close()
        self.remove_component(scanner_config['name'])
        self.scanner = Scanner(scanner_config, daq_config, self.temp, self.ureg)
        self.add_component(self.scanner)
        log.info('Scanner successfully added to microscope.')
    
    def _add_SQUID(self):
        """Add SQUID instrument to microscope.
        """
        squid_config = self.config['SQUID']
        if hasattr(self, 'SQUID'):
            #self.SQUID.clear_instances()
            self.SQUID.close()
        self.remove_component(squid_config['name'])
        squid_type = squid_config['type'].lower().capitalize()
        self.SQUID = getattr(sys.modules['squids'], squid_type)(squid_config)
        self.add_component(self.SQUID)
        log.info('{}(SQUID) successfully added to microscope.'.format(squid_type))
        
    def _add_lockins(self):
        """Add lockins to microscope.
        """
        for lockin, lockin_info in self.config['instruments']['lockins'].items():
            name = '{}_lockin'.format(lockin)
            address = lockin_info['address']
            if hasattr(self, name):
                #getattr(self, name, 'clear_instances')()
                getattr(self, name, 'close')()
            self.remove_component(name)
            instr = SR830(name, address, metadata={lockin: lockin_info})
            setattr(self, name, instr)
            self.add_component(getattr(self, '{}_lockin'.format(lockin)))
            log.info('{} successfully added to microscope.'.format(name))
            

    def ramp_keithley_volt(self, new_volt: float, npts: int, name: str, gate_speed:Optional[str]=None, quiet: Optional[bool]=False):

        #if name == 'ke2410_t':
            #obj = self.ke2410_t
        #elif name == 'ke2410_b':
            #obj = self.ke2410_b
        #elif name == 'ke2400_t':
            #obj = self.ke2400_t
        #elif name == 'ke2400_b':
            #obj = self.ke2400_b
        Kei = getattr(self, name)

        old_volt = Kei.volt()
        gate_speed = self.Q_(gate_speed).to('V/s').magnitude
        ramp_time = np.abs(old_volt - new_volt)/gate_speed
        time_step = ramp_time/npts
        ramp = np.linspace(old_volt, new_volt, npts)
        if quiet == False:
            msg = 'start ramping keithley {} at {} V/s to {} V.'
            log.warning(msg.format(name, gate_speed, new_volt))
        for point in ramp:
            time.sleep(time_step)
            Kei.volt(point)
        if quiet == False:
            log.warning('gate ramp completed')

    def ramp_keithley_volt_old(self, new_volt: float, npts: int, gate_speed:Optional[str]=None, quiet: Optional[bool]=False):
        old_volt = self.ke2410.volt()
        gate_speed = self.Q_(gate_speed).to('V/s').magnitude
        ramp_time = np.abs(old_volt - new_volt)/gate_speed
        time_step = ramp_time/npts
        ramp = np.linspace(old_volt, new_volt, npts)
        if quiet == False:
            msg = 'start ramping keithley at {} V/s to {} V.'
            log.warning(msg.format(gate_speed, new_volt))
        for point in ramp:
            time.sleep(time_step)
            self.ke2410.volt(point)
        if quiet == False:
            log.warning('gate ramp completed')
        

    def kei_line_gate(self, V_start: Any, V_end: Any, dV: Any, delay: Any, npts: int, gate_speed:Optional[str]=None):
        """make a line gate with keithely 2410
        Args:
            gate_grids: Dict of {axis_name: axis_meshgrid} from utils.make_gate_grids().
        """

        # set wait time and step
        m.ke2410.volt.set_delay(delay)
        m.ke2410.volt.set_step(dV)
        # set to start voltage
        self.ramp_keithley_volt(V_start, npt, gate_speed)
        # set to end voltage
        m.ke2410.volt.set(V_end)
        meas = qc.Measurement(exp=experiment)
        meas.register_parameter(m.ke2410.sense.sweep)
        with meas.run() as datasaver:
            datasaver.add_result((keithley.source.sweep_axis, keithley.source.sweep_axis()),
                         (keithley.sense.sweep, keithley.sense.sweep()))

            dataid = datasaver.run_id


    def set_lockins(self, measurement: Dict[str, Any]) -> None:
        """Initialize lockins for given measurement.
        Args:
            measurement: Dict of measurement parameters as defined
                in measurement configuration file.
        """
        channels = measurement['channels']
        for ch in channels:
            if 'lockin' in channels[ch]:
                lockin = '{}_lockin'.format(channels[ch]['lockin']['name'])
                for param in channels[ch]['lockin']:
                    if param in{'ch1_display', 'ch2_display'}:
                        parameters = getattr(self, lockin).parameters
                        value = channels[ch]['lockin'][param]
                        parameters[param].set(value)
                        log.info('Setting {} on {} to {}.'.format(param, lockin, value))
                        #print(parameters)

                    elif param != 'name':
                        parameters = getattr(self, lockin).parameters
                        unit = parameters[param].unit
                        value = self.Q_(channels[ch]['lockin'][param]).to(unit).magnitude
                        log.info('Setting {} on {} to {} {}.'.format(param, lockin, value, unit))
                        parameters[param].set(value)
        time.sleep(1)


    def gated_IV(self, gate_params: Dict[str, Any], update_snap: bool=True) -> Tuple[Any]:
        """Performs a two prob measurement at various gate voltage from a Keithley 
        Args:
            gate_params: Dict of gate parameters as defined in measurement config file
                in measurement configuration file.
            name: name of the keithley in use, 2400/2410, top/bottom
            direction: forward or backward sweep. Forward is increasing gate
            update_snap: Whether to update the microscope snapshot. Default True.
                (You may want this to be False when getting a plane or approaching.)
        Returns:
            Tuple[qcodes.DataSet, plots.TDCPlot]: data, tdc_plot
                DataSet and plot generated by the touchdown Loop.
        """
        daq_config = self.config['instruments']['daq']
        daq_name = daq_config['name']
        ai_channels = daq_config['channels']['analog_inputs']
        meas_channels = gate_params['channels']
        constants = gate_params['constants']
        channels = {} 
        for ch in meas_channels:
            channels.update({ch: ai_channels[ch]})
        daq_rate = self.Q_(daq_config['rate']).to('Hz').magnitude

        self.set_lockins(gate_params)
        self.snapshot(update=update_snap)
        dV = self.Q_(gate_params['dV']).to('V').magnitude
        name = gate_params['name']
        direction = gate_params['direction']
        if direction == 'forwards':
            startV, endV = sorted([self.Q_(lim).to('V').magnitude for lim in gate_params['range']])
        elif direction == 'backwards':
            endV, startV = sorted([self.Q_(lim).to('V').magnitude for lim in gate_params['range']])
        else:
            startV, endV = sorted([self.Q_(lim).to('V').magnitude for lim in gate_params['range']])
        delay = constants['wait_factor'] * self.SUSC_lockin.time_constant()
        prefactors = self.get_prefactors(gate_params)
        #: get channel prefactors in string form so they can be saved in metadata
        prefactor_strs = {}
        for ch, prefac in prefactors.items():
            unit = gate_params['channels'][ch]['unit']
            pre = prefac.to('{}/V'.format(unit))
            prefactor_strs.update({ch: '{} {}'.format(pre.magnitude, pre.units)})
        ai_task =  nidaqmx.Task('gated_IV_ai_task')
        self.remove_component('daq_ai')
        if hasattr(self, 'daq_ai'):
            self.daq_ai.clear_instances()
            self.daq_ai.close()
        self.daq_ai = DAQAnalogInputs('daq_ai', daq_name, daq_rate, channels, ai_task)
        loop_counter = utils.Counter()
        gate_plot = gatePlot(gate_params, self.ureg)
        gate_speed = gate_params['ramp_rate']
        npts = gate_params['ramp_points'] 
        #self.ke2410.volt(0)
        #if name == 'ke2410_t':
            #obj = self.ke2410_t
        #elif name == 'ke2410_b':
            #obj = self.ke2410_b
        #elif name == 'ke2400_t':
            #obj = self.ke2400_t
        #elif name == 'ke2400_b':
            #obj = self.ke2400_b  
        obj = getattr(self, name) 
        obj.output(1)
        self.ramp_keithley_volt(startV, npts, name, gate_speed) 
        time.sleep(1)
        loop = qc.Loop(obj.volt.sweep(startV, endV, dV), delay = delay
            ).each(
                #qc.Task(time.sleep, delay),
                self.daq_ai.voltage,
                qc.Task(self.scanner.get_gate, gate_plot, qc.loops.active_data_set, loop_counter),
                qc.Task(loop_counter.advance)
            ).then(
                qc.Task(ai_task.stop),
                qc.Task(ai_task.close),
                qc.Task(gate_plot.fig.show),
                qc.Task(gate_plot.save)

            )
        #print('hello')
        #: loop.metadata will be saved in DataSet
        loop.metadata.update(gate_params)
        loop.metadata.update({'prefactors': prefactor_strs})
        for idx, ch in enumerate(meas_channels):
            loop.metadata['channels'][ch].update({'idx': idx})
        data = loop.get_data_set(name=gate_params['fname'], write_period=None)
        try:
            log.info('Starting gating sample')
            loop.run()
        except KeyboardInterrupt:
            log.warning('gating interrupted by user. setting gate back to 0')
            #: Set break_loop = True so that get_plane() and approach() will be aborted
            #: Stop 'td_cap_ai_task' so that we can read our current position
            ai_task.stop()
            ai_task.close()
            #self.ramp_keithley_volt(0, npts, gate_speed)
            #self.ke2410.volt(0)
            #self.scanner.apply_gate(0)
            #self.CAP_lockin.amplitude(0.004)
            gate_plot.fig.show()
            gate_plot.save()
            log.info('Measurement aborted by user. DataSet saved to {}.'.format(data.location))
        #self.ramp_keithley_volt(0, npts, gate_speed)
        #self.ke2410.volt(0)
        #self.scanner.apply_gate(0)
        utils.gate_to_mat_file(data, real_units=True)
        return data, gate_plot

    def gated_IV_old(self, gate_params: Dict[str, Any], update_snap: bool=True) -> Tuple[Any]:
            """Performs a two prob measurement at various gate voltage from a Keithley 
            Args:
                gate_params: Dict of gate parameters as defined in measurement config file
                    in measurement configuration file.
                update_snap: Whether to update the microscope snapshot. Default True.
                    (You may want this to be False when getting a plane or approaching.)
            Returns:
                Tuple[qcodes.DataSet, plots.TDCPlot]: data, tdc_plot
                    DataSet and plot generated by the touchdown Loop.
            """
            daq_config = self.config['instruments']['daq']
            daq_name = daq_config['name']
            ai_channels = daq_config['channels']['analog_inputs']
            meas_channels = gate_params['channels']
            constants = gate_params['constants']
            channels = {} 
            for ch in meas_channels:
                channels.update({ch: ai_channels[ch]})
            daq_rate = self.Q_(daq_config['rate']).to('Hz').magnitude

            self.set_lockins(gate_params)
            self.snapshot(update=update_snap)
            dV = self.Q_(gate_params['dV']).to('V').magnitude
            startV, endV = sorted([self.Q_(lim).to('V').magnitude for lim in gate_params['range']])
            delay = constants['wait_factor'] * self.SUSC_lockin.time_constant()
            prefactors = self.get_prefactors(gate_params)
            #: get channel prefactors in string form so they can be saved in metadata
            prefactor_strs = {}
            for ch, prefac in prefactors.items():
                unit = gate_params['channels'][ch]['unit']
                pre = prefac.to('{}/V'.format(unit))
                prefactor_strs.update({ch: '{} {}'.format(pre.magnitude, pre.units)})
            ai_task =  nidaqmx.Task('gated_IV_ai_task')
            self.remove_component('daq_ai')
            if hasattr(self, 'daq_ai'):
                self.daq_ai.clear_instances()
                self.daq_ai.close()
            self.daq_ai = DAQAnalogInputs('daq_ai', daq_name, daq_rate, channels, ai_task)
            loop_counter = utils.Counter()
            gate_plot = gatePlot(gate_params, self.ureg)
            gate_speed = gate_params['ramp_rate']
            npts = gate_params['ramp_points'] 
            #self.ke2410.volt(0)
            self.ke2410.output(1)
            self.ramp_keithley_volt(startV, npts, gate_speed) 
            time.sleep(3)
            loop = qc.Loop(self.ke2410.volt.sweep(startV, endV, dV)
                ).each(
                    qc.Task(time.sleep, delay),
                    self.daq_ai.voltage,
                    qc.Task(self.scanner.get_gate, gate_plot, qc.loops.active_data_set, loop_counter),
                    qc.Task(loop_counter.advance)
                ).then(
                    qc.Task(ai_task.stop),
                    qc.Task(ai_task.close),
                    qc.Task(gate_plot.fig.show),
                    qc.Task(gate_plot.save)

                )
            #print('hello')
            #: loop.metadata will be saved in DataSet
            loop.metadata.update(gate_params)
            loop.metadata.update({'prefactors': prefactor_strs})
            for idx, ch in enumerate(meas_channels):
                loop.metadata['channels'][ch].update({'idx': idx})
            data = loop.get_data_set(name=gate_params['fname'], write_period=None)
            try:
                log.info('Starting gating sample')
                loop.run()
            except KeyboardInterrupt:
                log.warning('gating interrupted by user. setting gate back to 0')
                #: Set break_loop = True so that get_plane() and approach() will be aborted
                #: Stop 'td_cap_ai_task' so that we can read our current position
                ai_task.stop()
                ai_task.close()
                #self.ramp_keithley_volt(0, npts, gate_speed)
                #self.ke2410.volt(0)
                #self.scanner.apply_gate(0)
                #self.CAP_lockin.amplitude(0.004)
                gate_plot.fig.show()
                gate_plot.save()
                log.info('Measurement aborted by user. DataSet saved to {}.'.format(data.location))
            #self.ramp_keithley_volt(0, npts, gate_speed)
            #self.ke2410.volt(0)
            #self.scanner.apply_gate(0)
            utils.gate_to_mat_file(data, real_units=True)
            return data, gate_plot


    def double_gated_IV(self, gate_params: Dict[str, Any], update_snap: bool=True) -> Tuple[Any]:
        """Performs a two prob measurement at various gate voltage from a Keithley 
        Args:
            gate_params: Dict of gate parameters as defined in measurement config file
                in measurement configuration file.
            update_snap: Whether to update the microscope snapshot. Default True.
                (You may want this to be False when getting a plane or approaching.)
        Returns:
            Tuple[qcodes.DataSet, plots.TDCPlot]: data, tdc_plot
                DataSet and plot generated by the touchdown Loop.
        """

        # take in all relavent constants
        daq_config = self.config['instruments']['daq']
        daq_name = daq_config['name']
        daq_rate = self.Q_(daq_config['rate']).to('Hz').magnitude
        ai_channels = daq_config['channels']['analog_inputs']
        ao_channels = daq_config['channels']['analog_outputs']
        meas_channels = gate_params['channels']
        constants = gate_params['constants']

        # set up channels to be measured 
        channels = {} 
        for ch in meas_channels:
            channels.update({ch: ai_channels[ch]})  
        nchannels = len(channels.keys())

        # set up x y axis of maps
        dVX = self.Q_(gate_params['dVX']).to('V').magnitude
        dVY = self.Q_(gate_params['dVY']).to('V').magnitude
        VXstartV, VXendV = sorted([self.Q_(lim).to('V').magnitude for lim in gate_params['VXrange']])
        VYstartV, VYendV = sorted([self.Q_(lim).to('V').magnitude for lim in gate_params['VYrange']])

        # set up fast and slow axis and Keithley
        xkey = getattr(self, gate_params['Xkei']) 
        ykey = getattr(self, gate_params['Ykei']) 
        xkey.output(1)
        ykey.output(1)
        fast_ax = gate_params['fast_ax']
        gate_speed = gate_params['ramp_rate']
        npts = gate_params['ramp_points'] 
        if fast_ax == 'X':
            #pix_per_line = int((VYendV - VYstartV)/dVY) + 1 
            slow_ax = 'Y'
            fastKeyString = gate_params['Xkei']
            slowKeyString = gate_params['Ykei']
            fastKey = xkey
            slowKey = ykey
            startV = VXstartV 
            endV = VXendV 
            dV = dVX
            line_startV = VYstartV
            line_endV = VYendV
            line_dV = dVY
        else:
            #pix_per_line = int((VXendV - VXstartV)/dVX) + 1
            slow_ax = 'X'
            fastKeyString = gate_params['Ykei']
            slowKeyString = gate_params['Xkei']
            fastKey = ykey
            slowKey = xkey
            startV = VYstartV 
            endV = VYendV 
            dV = dVY
            line_startV = VXstartV
            line_endV = VXendV
            line_dV = dVX
        self.ramp_keithley_volt(startV, npts, fastKeyString, gate_speed) 
        self.ramp_keithley_volt(line_startV, npts, slowKeyString, gate_speed) 

        delay = constants['wait_factor'] * self.SUSC_lockin.time_constant() *self.ureg('s')
        #delay = constants['wait_factor'] * self.SUSC_lockin.time_constant()
        line_delay = constants['line_wait_factor'] * self.SUSC_lockin.time_constant()
        sample_time = constants['sample_time'] 
        sample_pts = int(sample_time * daq_rate)
        #line_duration = pix_per_line * delay
        #pts_per_line = int(daq_rate * line_duration.to('s').magnitude)
        #gate_vectors = utils.make_gate_vectors(gate_params, self.ureg)
        #gate_grids = utils.make_gate_grids(gate_vectors, pts_per_line)
        self.set_lockins(gate_params)
        self.snapshot(update=update_snap)
        prefactors = self.get_prefactors(gate_params)


        #: get channel prefactors in string form so they can be saved in metadata
        prefactor_strs = {}
        for ch, prefac in prefactors.items():
            unit = gate_params['channels'][ch]['unit']
            pre = prefac.to('{}/V'.format(unit))
            prefactor_strs.update({ch: '{} {}'.format(pre.magnitude, pre.units)})

        # set up DAQ
        ai_task =  nidaqmx.Task('double_gated_IV_ai_task')
        self.remove_component('daq_ai')
        if hasattr(self, 'daq_ai'):
            self.daq_ai.clear_instances()
            self.daq_ai.close()
        self.daq_ai = DAQAnalogInputs('daq_ai', daq_name, daq_rate, channels, ai_task,
                                      samples_to_read=sample_pts, target_points=1)
        self.add_component(self.daq_ai)
        double_gate_plot = DoubleGatedPlot(gate_params, self.ureg) 

        #self.ke2410.volt(0)

        

        # DAQ output is fast axis
        loop = qc.Loop(slowKey.volt.sweep(line_startV, line_endV, line_dV), delay = line_delay
            ).each(
            qc.Task(self.ramp_keithley_volt, startV, npts, fastKeyString, gate_speed, True),
            #qc.Task(double_gate_plot.update, qc.loops.active_data_set),
            #qc.Task(double_gate_plot.save),
            qc.loops.Loop(fastKey.volt.sweep(startV, endV, dV), delay = self.Q_(delay).to('s').magnitude).each(
                #start AI task
                qc.Task(ai_task.start),
                #: Acquire voltage from all active AI channels
                self.daq_ai.voltage,
                # wait until  done
                qc.Task(ai_task.wait_until_done, timeout=10),
                # stop ai task when it is done
                qc.Task(ai_task.stop),
                #qc.Task(print,datetime.now),
                #: Update and save plot
                #qc.Task(double_gate_plot.update, qc.loops.active_data_set),
                #qc.Task(double_gate_plot.save)
                #qc.Task(double_gate_plot.update, qc.loops.active_data_set, loop_counter),
                #qc.Task(double_gate_plot.save),
                #qc.Task(loop_counter.advance)
                #qc.Task(ai_task.close),
                #self.daq_ai.clear_instances
                #qc.Task(self.daq_ai.clear_instances),
                #qc.Task(nidaqmx.Task('double_gated_IV_ai_task'), ai_task)
                ),
            qc.Task(double_gate_plot.update, qc.loops.active_data_set),
            qc.Task(double_gate_plot.save)
            ).then(
                qc.Task(ai_task.stop),
                qc.Task(ai_task.close),
                qc.Task(double_gate_plot.fig.show),
                qc.Task(double_gate_plot.save))

        #: loop.metadata will be saved in DataSet
        loop.metadata.update(gate_params)
        loop.metadata.update({'prefactors': prefactor_strs})
        for idx, ch in enumerate(meas_channels):
            loop.metadata['channels'][ch].update({'idx': idx})
        data = loop.get_data_set(name=gate_params['fname'], write_period=None)

        try:
            log.info('Starting gating sample')
            loop.run()
        except KeyboardInterrupt:
            log.warning('gating interrupted by user. setting gate back to 0')
            #: Set break_loop = True so that get_plane() and approach() will be aborted
            #: Stop 'td_cap_ai_task' so that we can read our current position
            ai_task.stop()
            ai_task.close()
            self.daq_ai.clear_instances()
            #self.ramp_keithley_volt(0, npts, gate_speed)
            #self.ke2410.output(0)
            #self.CAP_lockin.amplitude(0.004)
            double_gate_plot.fig.show()
            double_gate_plot.save()
            log.info('Measurement aborted by user. DataSet saved to {}.'.format(data.location))
        utils.double_gate_to_mat_file(data, real_units=True)
        # turn off both gates
        #self.ramp_keithley_volt(0, npts, gate_speed)
        #self.scanner.apply_gate(0, gate_speed=gate_speed)
        return data, double_gate_plot

    def double_gated_IV_old(self, gate_params: Dict[str, Any], update_snap: bool=True) -> Tuple[Any]:
        """Performs a two prob measurement at various gate voltage from a Keithley 
        Args:
            gate_params: Dict of gate parameters as defined in measurement config file
                in measurement configuration file.
            update_snap: Whether to update the microscope snapshot. Default True.
                (You may want this to be False when getting a plane or approaching.)
        Returns:
            Tuple[qcodes.DataSet, plots.TDCPlot]: data, tdc_plot
                DataSet and plot generated by the touchdown Loop.
        """

        # take in all relavent constants
        daq_config = self.config['instruments']['daq']
        daq_name = daq_config['name']
        daq_rate = self.Q_(daq_config['rate']).to('Hz').magnitude
        ai_channels = daq_config['channels']['analog_inputs']
        ao_channels = daq_config['channels']['analog_outputs']
        meas_channels = gate_params['channels']
        constants = gate_params['constants']

        # set up channels to be measured 
        channels = {} 
        for ch in meas_channels:
            channels.update({ch: ai_channels[ch]})    
        nchannels = len(channels.keys())

        # set up x y axis of maps
        dVT = self.Q_(gate_params['dVT']).to('V').magnitude
        dVB = self.Q_(gate_params['dVB']).to('V').magnitude
        VTstartV, VTendV = sorted([self.Q_(lim).to('V').magnitude for lim in gate_params['VTrange']])
        VBstartV, VBendV = sorted([self.Q_(lim).to('V').magnitude for lim in gate_params['VBrange']])

        # set up fast and slow axis
        fast_ax = gate_params['DAQ_AO']
        slow_ax = 'B' if fast_ax == 'T' else 'T'
        if fast_ax == 'T':
            pix_per_line = int((VTendV - VTstartV)/dVT) + 1 
            startV = VBstartV 
            endV = VBendV 
            dV = dVB
            line_startV = VTstartV
            line_endV = VTendV
        else:
            pix_per_line = int((VBendV - VBstartV)/dVB) + 1
            startV = VTstartV 
            endV = VTendV 
            dV = dVT
            line_startV = VBstartV
            line_endV = VTendV
        delay = constants['wait_factor'] * self.SUSC_lockin.time_constant() *self.ureg('s')
        line_delay = constants['line_wait_factor'] * self.SUSC_lockin.time_constant()
        line_duration = pix_per_line * delay
        pts_per_line = int(daq_rate * line_duration.to('s').magnitude)
        gate_vectors = utils.make_gate_vectors(gate_params, self.ureg)
        gate_grids = utils.make_gate_grids(gate_vectors, slow_ax, fast_ax,
                                           pts_per_line)
        self.set_lockins(gate_params)
        self.snapshot(update=update_snap)
        prefactors = self.get_prefactors(gate_params)

        #: get channel prefactors in string form so they can be saved in metadata
        prefactor_strs = {}
        for ch, prefac in prefactors.items():
            unit = gate_params['channels'][ch]['unit']
            pre = prefac.to('{}/V'.format(unit))
            prefactor_strs.update({ch: '{} {}'.format(pre.magnitude, pre.units)})

        ai_task =  nidaqmx.Task('double_gated_IV_ai_task')
        self.remove_component('daq_ai')
        if hasattr(self, 'daq_ai'):
            self.daq_ai.clear_instances()
            self.daq_ai.close()
        self.daq_ai = DAQAnalogInputs('daq_ai', daq_name, daq_rate, channels, ai_task,
                                      samples_to_read=pts_per_line, target_points=pix_per_line,
                                      #: Very important to synchronize AOs and AIs
                                      clock_src='ao/SampleClock', timeout=200)
        self.add_component(self.daq_ai)

        loop_counter = utils.Counter()
        double_gate_plot = DoubleGatedPlot(gate_params, self.ureg) 
        gate_speed = gate_params['ramp_rate']
        npts = gate_params['ramp_points'] 
        #self.ke2410.volt(0)
        self.ke2410.output(1)
        self.ramp_keithley_volt(startV, npts, gate_speed)


        # DAQ output is fast axis
        loop = qc.Loop(self.ke2410.volt.sweep(startV, endV, dV)
            ).each(
                # pause between lines
                qc.Task(time.sleep, line_delay),
                # apply target voltage
                qc.Task(self.scanner.apply_gate, line_startV, None, True),
                # create AO task
                qc.Task(self.scanner.DAQ_line_gate, gate_grids, ao_channels, daq_rate, loop_counter),
                # start AI task               
                qc.Task(ai_task.start),
                #: Start AO task
                qc.Task(self.scanner.control_ao_task, 'start'),
                #: Acquire voltage from all active AI channels
                self.daq_ai.voltage,
                # wait until both are done
                qc.Task(ai_task.wait_until_done, timeout=600),
                qc.Task(self.scanner.control_ao_task, 'wait_until_done'),
                # stop ai task when it is done
                qc.Task(ai_task.stop),
                # stop and close ao task for the next line
                qc.Task(self.scanner.control_ao_task, 'stop'),
                qc.Task(self.scanner.control_ao_task, 'close'),
                #: Update and save plot
                qc.Task(double_gate_plot.update, qc.loops.active_data_set, loop_counter),
                qc.Task(double_gate_plot.save),
                qc.Task(loop_counter.advance)

            ).then(
                qc.Task(ai_task.stop),
                qc.Task(ai_task.close),
                qc.Task(double_gate_plot.fig.show),
                qc.Task(double_gate_plot.save)
            )

        #: loop.metadata will be saved in DataSet
        loop.metadata.update(gate_params)
        loop.metadata.update({'prefactors': prefactor_strs})
        for idx, ch in enumerate(meas_channels):
            loop.metadata['channels'][ch].update({'idx': idx})
        data = loop.get_data_set(name=gate_params['fname'], write_period=None)

        try:
            log.info('Starting gating sample')
            loop.run()
        except KeyboardInterrupt:
            log.warning('gating interrupted by user. setting gate back to 0')
            #: Set break_loop = True so that get_plane() and approach() will be aborted
            #: Stop 'td_cap_ai_task' so that we can read our current position
            ai_task.stop()
            ai_task.close()
            #self.ramp_keithley_volt(0, npts, gate_speed)
            #self.ke2410.output(0)
            #self.CAP_lockin.amplitude(0.004)
            double_gate_plot.fig.show()
            double_gate_plot.save()
            log.info('Measurement aborted by user. DataSet saved to {}.'.format(data.location))
        utils.double_gate_to_mat_file(data, real_units=True)
        # turn off both gates
        #self.ramp_keithley_volt(0, npts, gate_speed)
        #self.scanner.apply_gate(0, gate_speed=gate_speed)
        return data, double_gate_plot


    def Temp_4probe(self, t4p_params: Dict[str, Any], update_snap: bool=True) -> Tuple[Any]:
        """Performs a four-probe measurement at various temperature 
        Args:
            t4p_params: Dict of temperature and 4-probe method parameters as defined in measurement config file
                in measurement configuration file.
            update_snap: Whether to update the microscope snapshot. Default True.
                (You may want this to be False when getting a plane or approaching.)
        Returns:
            Tuple[qcodes.DataSet, plots.t4pPlot]: data, t4p_plot
                DataSet and plot generated by the IV Loop.
        """
        daq_config = self.config['instruments']['daq']
        daq_name = daq_config['name']
        ai_channels = daq_config['channels']['analog_inputs']
        meas_channels = t4p_params['channels']
        Tramp_rate = self.Q_(t4p_params['T_ramp_rate']).to('K/min').magnitude
        constants = t4p_params['constants']
        channels = {} 
        for ch in meas_channels:
            channels.update({ch: ai_channels[ch]})
        daq_rate = self.Q_(daq_config['rate']).to('Hz').magnitude
        sampleduration = self.Q_(constants['sampleduration']).to('sec').magnitude
        nsamples = int(daq_rate * sampleduration)

        self.set_lockins(t4p_params)
        self.snapshot(update=update_snap)
        #: delta T
        dT = self.Q_(t4p_params['dT']).to('K').magnitude
        #: temperature range
        self.ls372.ramp_rate(Tramp_rate)
        ramp_rate = self.ls372.ramp_rate()
        ramp_rate = ramp_rate.split(',')[1]
        ramp_rate = float(ramp_rate)

        dV = self.Q_(t4p_params['dV']).to('V').magnitude
        delay = constants['wait_factor'] * self.SUSC_lockin.time_constant()
        T_delay = constants['T_wait_factor'] * abs(dT)/ramp_rate * 60 + delay

        prefactors = self.get_prefactors(t4p_params)
        #: get channel prefactors in string form so they can be saved in metadata
        prefactor_strs = {}
        for ch, prefac in prefactors.items():
            unit = t4p_params['channels'][ch]['unit']
            pre = prefac.to('{}/V'.format(unit))
            prefactor_strs.update({ch: '{} {}'.format(pre.magnitude, pre.units)})
        ai_task =  nidaqmx.Task('temp_4probe_ai_task')
        self.remove_component('daq_ai')
        if hasattr(self, 'daq_ai'):
            self.daq_ai.clear_instances()
            self.daq_ai.close()
        self.daq_ai = DAQAnalogInputs('daq_ai', daq_name, daq_rate, channels, ai_task,samples_to_read=nsamples,target_points=1, timeout=sampleduration+1)
        loop_counter = utils.Counter()

        # set up x y axis of maps
        IstartV, IendV = sorted([self.Q_(lim).to('V').magnitude for lim in t4p_params['V_range']])
        TstartV, TendV = sorted([self.Q_(lim).to('V').magnitude for lim in t4p_params['T_range']])

        # set up fast and slow axis
        fast_ax = t4p_params['fast_ax']
        slow_ax = 'T' if fast_ax == 'I' else 'I'
        if fast_ax == 'I':
            pix_per_line = int((IendV - IstartV)/dV) + 1 
            startV = TstartV 
            endV = TendV 
            dV = dV
            line_startV = IstartV
            line_endV = IendV
        else:
            pix_per_line = int((TendV - TstartV)/dT) + 1
            startV = IstartV 
            endV = IendV 
            dV = dT
            line_startV = TstartV
            line_endV = TendV
        line_delay = constants['line_wait_factor'] * self.SUSC_lockin.time_constant()
        line_duration = pix_per_line * delay
        pts_per_line = int(daq_rate * line_duration.to('s').magnitude)
        t4p_vectors = utils.make_t4p_vectors(t4p_params, self.ureg)
        t4p_grids = utils.make_t4p_grids(t4p_vectors, slow_ax, fast_ax,
                                           pts_per_line)

        t4p_plot = t4pPlot(t4p_params, self.ureg)
        LIA_speed = t4p_params['V_ramp_rate']
        npts = t4p_params['V_ramp_points'] 

        if fast_ax == 'I':
            loop = qc.Loop(self.ls372.set_temperature.sweep(startT, endT, dT)
            ).each(
                # pause between lines
                qc.Task(time.sleep, line_delay),
                # apply target voltage
                qc.Task(self.scanner.LIA, line_startV, None, True),
                # create AO tast
                qc.Task(self.scanner.DAQ_line_gate, gate_grids, ao_channels, daq_rate, loop_counter),
                # start AI task               
                qc.Task(ai_task.start),
                #: Start AO task
                qc.Task(self.scanner.control_ao_task, 'start'),
                #: Acquire voltage from all active AI channels
                self.daq_ai.voltage,
                # wait until both are done
                qc.Task(ai_task.wait_until_done, timeout=200),
                qc.Task(self.scanner.control_ao_task, 'wait_until_done'),
                # stop ai task when it is done
                qc.Task(ai_task.stop),
                # stop and close ao task for the next line
                qc.Task(self.scanner.control_ao_task, 'stop'),
                qc.Task(self.scanner.control_ao_task, 'close'),
                #: Update and save plot
                qc.Task(double_gate_plot.update, qc.loops.active_data_set, loop_counter),
                qc.Task(double_gate_plot.save),
                qc.Task(loop_counter.advance)

            ).then(
                qc.Task(ai_task.stop),
                qc.Task(ai_task.close),
                qc.Task(double_gate_plot.fig.show),
                qc.Task(double_gate_plot.save)
            )
        else:


            self.ke2400.volt(0)
            self.ke2400.output(1)
            self.ramp_keithley_volt(startV, npts, LIA_speed) 
            loop = qc.Loop(self.ke2400.volt.sweep(startV, endV, dV)
                ).each(
                    qc.Task(time.sleep, delay),
                    self.daq_ai.voltage,
                    qc.Task(self.scanner.get_tdidv, tempdidv_plot, qc.loops.active_data_set, loop_counter),
                    qc.Task(loop_counter.advance)
                ).then(
                    qc.Task(ai_task.stop),
                    qc.Task(ai_task.close),
                    qc.Task(tempdidv_plot.fig.show),
                    qc.Task(tempdidv_plot.save)

                )
            #: loop.metadata will be saved in DataSet
            loop.metadata.update(tempdidv_params)
            loop.metadata.update({'prefactors': prefactor_strs})
        for idx, ch in enumerate(meas_channels):
            loop.metadata['channels'][ch].update({'idx': idx})
        data = loop.get_data_set(name=tempdidv_params['fname'], write_period=None)
        try:
            log.info('Starting measuring dIdV on sample')
            loop.run()
        except KeyboardInterrupt:
            log.warning('measuring interrupted by user. setting gate back to 0')
            #: Set break_loop = True so that get_plane() and approach() will be aborted
            #: Stop 'td_cap_ai_task' so that we can read our current position
            ai_task.stop()
            ai_task.close()
            self.ramp_keithley_volt(0, npts, LIA_speed)
            self.ke2400.volt(0)
            self.scanner.apply_gate(0)
            #self.CAP_lockin.amplitude(0.004)
            tempdidv_plot.fig.show()
            tempdidv_plot.save()
            log.info('Measurement aborted by user. DataSet saved to {}.'.format(data.location))
        self.ramp_keithley_volt(0, npts, LIA_speed)
        self.ke2400.volt(0)
        self.scanner.apply_gate(0)
        utils.tdidv_to_mat_file(data, real_units=True)
        return data, tempdidv_plot



    def temp_series(self, temps_params: Dict[str, Any], update_snap: bool=True) -> Tuple[Any]:
        """Performs MAG and SUSCX measurements over a range of temperature

        Args:
            temps_params: Dict of temperature range and scan parameters as defined
                in measurement configuration file.
            update_snap: Whether to update the microscope snapshot. Default True.
                (You may want this to be False when getting a plane or approaching.)

        Returns:
            Tuple[qcodes.DataSet, plots.temp_series]: data, MSvTPlot
                DataSet and plot.
        """
        daq_config = self.config['instruments']['daq']
        daq_name = daq_config['name']
        ai_channels = daq_config['channels']['analog_inputs']
        meas_channels = temps_params['channels']
        Tramp_rate = self.Q_(temps_params['ramp_rate']).to('K/min').magnitude
        constants = temps_params['constants']
        channels = {} 
        for ch in meas_channels:
            channels.update({ch: ai_channels[ch]})
        # daq_rate = self.Q_(daq_config['rate']).to('Hz').magnitude
        daq_rate = self.Q_(constants['daq_rate']).to('Hz').magnitude
        sampleduration = self.Q_(constants['sampleduration']).to('sec').magnitude
        nsamples = int(daq_rate * sampleduration)
        

        self.set_lockins(temps_params)
        self.snapshot(update=update_snap)
        #: delta T
        dT = self.Q_(temps_params['dT']).to('K').magnitude
        #: temperature range
        self.ls372.ramp_rate(Tramp_rate)
        ramp_rate = self.ls372.ramp_rate()
        ramp_rate = ramp_rate.split(',')[1]
        ramp_rate = float(ramp_rate)
        startT, endT = [self.Q_(lim).to('K').magnitude for lim in temps_params['range']]


        delay = constants['wait_factor'] * self.SUSC_lockin.time_constant()
        T_delay = constants['T_wait_factor'] * abs(dT)/ramp_rate * 60 + delay
        # print("data acquization time interval is(s)")
        # print(T_delay)

        prefactors = self.get_prefactors(temps_params)
        #: get channel prefactors in string form so they can be saved in metadata
        prefactor_strs = {}
        for ch, prefac in prefactors.items():
            unit = temps_params['channels'][ch]['unit']
            pre = prefac.to('{}/V'.format(unit))
            prefactor_strs.update({ch: '{} {}'.format(pre.magnitude, pre.units)})
        ai_task =  nidaqmx.Task('MAG_SUSCX_v_T_ai_task')
        self.remove_component('daq_ai')
        if hasattr(self, 'daq_ai'):
            self.daq_ai.clear_instances()
            self.daq_ai.close()
        # self.daq_ai = DAQAnalogInputs('daq_ai', daq_name, daq_rate, channels, ai_task)
        self.daq_ai = DAQAnalogInputs('daq_ai', daq_name, daq_rate, channels, ai_task,samples_to_read=nsamples,target_points=1, timeout=sampleduration+1)
        loop_counter = utils.Counter()
        MSvT_plot = MSvTPlot(temps_params, self.ureg) 
        loop = qc.Loop(self.ls372.set_temperature.sweep(startT, endT, dT)
            ).each(
                qc.Task(time.sleep, T_delay),
                self.daq_ai.voltage,
                qc.Task(self.scanner.get_MSvT, MSvT_plot, qc.loops.active_data_set, loop_counter),
                qc.Task(loop_counter.advance),
            ).then(
                qc.Task(ai_task.stop),
                qc.Task(ai_task.close),
                #qc.Task(self.CAP_lockin.amplitude, 0.004),
                #qc.Task(self.SUSC_lockin.amplitude, 0.004),
                qc.Task(MSvT_plot.fig.show),
                qc.Task(MSvT_plot.save)
            )
        #: loop.metadata will be saved in DataSet
        loop.metadata.update(temps_params)
        loop.metadata.update({'prefactors': prefactor_strs})
        for idx, ch in enumerate(meas_channels):
            loop.metadata['channels'][ch].update({'idx': idx})
        data = loop.get_data_set(name=temps_params['fname'], write_period=None)
        try:
            log.info('Starting temperature sweep:')
            loop.run()
        except KeyboardInterrupt:
            log.warning('Temperature series interrupted by user. DataSet saved to {}.'.format(data.location))
            #: Set break_loop = True so that get_plane() and approach() will be aborted
            self.scanner.break_loop = True
            ai_task.stop()
            ai_task.close()
        finally:
            try:
                #: Stop 'MAG_SUSCX_v_T_ai_task' so that we can read our current position
                ai_task.stop()
                ai_task.close()
                MSvT_plot.fig.show()
                MSvT_plot.save()
            except:
                pass
        utils.MSvT_to_mat_file(data, real_units=True)

        return data, MSvT_plot

 
    def remove_component(self, name: str) -> None:
        """Remove a component (instrument) from the microscope.
        Args:
            name: Name of component to remove.
        """
        if name in self.components:
            _ = self.components.pop(name)
            log.info('Removed {} from microscope.'.format(name))
        else:
            log.debug('Microscope has no component with the name {}'.format(name))  

#    def susc_temp()
