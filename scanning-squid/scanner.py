

import qcodes as qc
from qcodes.instrument.base import Instrument
import qcodes.utils.validators as vals
import utils
import time
from scipy import io
from scipy.interpolate import Rbf
from typing import Dict, List, Optional, Sequence, Any, Union
import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType, TaskMode
import logging
log = logging.getLogger(__name__)

class Scanner(Instrument):
    """Controls DAQ AOs to drive the scanner.
    """   
    def __init__(self, scanner_config: Dict[str, Any], daq_config: Dict[str, Any],
                 temp: str, ureg: Any, **kwargs) -> None:
        """
        Args:
            scanner_config: Scanner configuration dictionary as defined
                in microscope configuration JSON file.
            daq_config: DAQ configuration dictionary as defined
                in microscope configuration JSON file.
            temp: 'LT' or 'RT' - sets the scanner voltage limit for each axis
                based on temperature mode.
            ureg: pint UnitRegistry, manages units.
        """
        super().__init__(scanner_config['name'], **kwargs)
        if temp.upper() not in ['LT', 'RT']:
            raise ValueError('Temperature mode must be "LT" or "RT".')
        self.temp = temp.upper()
        self.ureg = ureg
        self.Q_ = ureg.Quantity
        self.metadata.update(scanner_config)
        self.metadata.update({'daq': daq_config})
        self._parse_unitful_quantities()
        self._initialize_parameters()
        #self.goto([0, 0, 0])
        
    def _parse_unitful_quantities(self):
        """Parse strings from configuration dicts into Quantities with units.
        """
        self.daq_rate = self.Q_(self.metadata['daq']['rate']).to('Hz').magnitude
        self.voltage_retract = {'RT': self.Q_(self.metadata['voltage_retract']['RT']),
                                'LT': self.Q_(self.metadata['voltage_retract']['LT'])}
        self.speed = self.Q_(self.metadata['speed']['value'])
        self.gate_speed = self.Q_(self.metadata['gate_speed']['value'])
        self.constants = {'comment': self.metadata['constants']['comment']}
        self.voltage_limits = {'RT': {},
                               'LT': {},
                               'unit': self.metadata['voltage_limits']['unit'],
                               'comment': self.metadata['voltage_limits']['comment']}
        unit = self.voltage_limits['unit']
        for axis in ['x', 'y', 'z']:
            self.constants.update({axis: self.Q_(self.metadata['constants'][axis])})
            for temp in ['RT', 'LT']:
                lims = [lim *self.ureg(unit) for lim in sorted(self.metadata['voltage_limits'][temp][axis])]
                self.voltage_limits[temp].update({axis: lims})
                
    def _initialize_parameters(self):
        """Add parameters to instrument upon initialization.
        """
        v_limits = []
        for axis in ['x', 'y', 'z']:
            lims = self.voltage_limits[self.temp][axis]
            lims_V = [lim.to('V').magnitude for lim in lims]
            v_limits += lims_V
        self.add_parameter('position',
                            label='Scanner position',
                            unit='V',
                            vals=vals.Lists(
                                elt_validator=vals.Numbers(min(v_limits), max(v_limits))),
                            get_cmd=self.get_pos,
                            set_cmd=self.goto
                            )
        for i, axis in enumerate(['x', 'y', 'z']):
            lims = self.voltage_limits[self.temp][axis]
            lims_V = [lim.to('V').magnitude for lim in lims]
            self.add_parameter('position{}'.format(axis),
                           label='{} position'.format(axis),
                           unit='V',
                           vals=vals.Numbers(min(lims_V), max(lims_V)),
                           get_cmd=(lambda idx=i: self.get_pos()[idx]),
                           set_cmd=getattr(self, '_goto_{}'.format(axis))
                           )
        

    def check_gate(self, axis: str) -> float:
        """Get current gate voltage.
        Returns:
            float of specified gate.
        """
        idx = self.metadata['daq']['channels']['analog_inputs'][axis]
        channel = self.metadata['daq']['name'] + '/ai{}'.format(idx)    
        with nidaqmx.Task('get_gate_ai_task') as ai_task:
            ai_task.ai_channels.add_ai_voltage_chan(channel, axis, min_val=-10, max_val=10)
            result_raw = np.round(ai_task.read(), decimals=3)
        result = 10 
        if result_raw < 10:
            result = result_raw
        #ai_task.close()
        return result

    

    
    def apply_gate(self, new_gate: float,
             gate_speed: Optional[str]=None, quiet: Optional[bool]=False) -> None:
        """apply a DC gate voltage from a DAC AO.
        Args:
            new_gate: a gate voltage one wants to apply through DAC's AO.

            gated_speed: Speed at which to change gate voltage (e.g. '2 V/s') in DAQ voltage units.
                Default set in microscope configuration JSON file.
            quiet: If True, only logs changes in logging.DEBUG mode.
                (goto is called many times during, e.g., a scan.) Default: False.
        """
        #ai_task.close()
        old_gate = self.check_gate('G')
        if gate_speed is None:
            gate_speed = self.gate_speed.to('V/s').magnitude
        else:
            gate_speed = self.Q_(gate_speed).to('V/s').magnitude

        gate_range = self.metadata['gate_range']
        if new_gate > gate_range[1] or new_gate < gate_range[0]:
                err = 'Requested gate is out of range. '
                err += 'gate range is {} V.'
                raise ValueError(err.format(gate_range))
 
        ramp = self.gate_ramp(old_gate, new_gate, gate_speed)
        with nidaqmx.Task('goto_ao_task') as ao_task:
            axis = 'G'
            idx = self.metadata['daq']['channels']['analog_outputs'][axis]
            channel = self.metadata['daq']['name'] + '/ao{}'.format(idx, axis)
            ao_task.ao_channels.add_ao_voltage_chan(channel)
            ao_task.timing.cfg_samp_clk_timing(self.daq_rate, samps_per_chan=len(ramp))
            pts = ao_task.write(ramp, auto_start=False, timeout=60)
            ao_task.start()
            ao_task.wait_until_done(timeout=60)
            log.debug('Wrote {} samples to {}.'.format(pts, ao_task.channel_names))
        current_gate = self.check_gate('G')
        if quiet:
            log.debug('Changed gate from {} V to {} V.'.format(old_gate, current_gate))
        else:
             log.info('Changed gate from {} V to {} V.'.format(old_gate, current_gate))

    def apply_gate_to_chan(self, new_gate: float, axis: str,
             gate_speed: Optional[str]=None, quiet: Optional[bool]=False) -> None:
        """apply a DC gate voltage from a DAC AO.
        Args:
            new_gate: a gate voltage one wants to apply through DAC's AO.

            gated_speed: Speed at which to change gate voltage (e.g. '2 V/s') in DAQ voltage units.
                Default set in microscope configuration JSON file.
            quiet: If True, only logs changes in logging.DEBUG mode.
                (goto is called many times during, e.g., a scan.) Default: False.
        """
        #ai_task.close()
        old_gate = self.check_gate(axis)
        if gate_speed is None:
            gate_speed = self.gate_speed.to('V/s').magnitude
        else:
            gate_speed = self.Q_(gate_speed).to('V/s').magnitude

        gate_range = self.metadata['gate_range']
        if new_gate > gate_range[1] or new_gate < gate_range[0]:
                err = 'Requested gate is out of range. '
                err += 'gate range is {} V.'
                raise ValueError(err.format(gate_range))
 
        ramp = self.gate_ramp(old_gate, new_gate, gate_speed)
        with nidaqmx.Task('goto_ao_task') as ao_task:
            #axis = axis
            idx = self.metadata['daq']['channels']['analog_outputs'][axis]
            channel = self.metadata['daq']['name'] + '/ao{}'.format(idx, axis)
            ao_task.ao_channels.add_ao_voltage_chan(channel)
            ao_task.timing.cfg_samp_clk_timing(self.daq_rate, samps_per_chan=len(ramp))
            pts = ao_task.write(ramp, auto_start=False, timeout=60)
            ao_task.start()
            ao_task.wait_until_done(timeout=60)
            log.debug('Wrote {} samples to {}.'.format(pts, ao_task.channel_names))
        current_gate = self.check_gate(axis)
        npt = len(ramp)
        if quiet:
            log.debug('Changed gate from {} V to {} V at {} V/s in {} points.'.format(old_gate, current_gate, gate_speed, npt))
        else:
             log.info('Changed gate from {} V to {} V at {} V/s in {} points.'.format(old_gate, current_gate, gate_speed, npt))



    def Kei_line_gate(self, gate_grids: Dict[str, np.ndarray], ao_channels: Dict[str, int],
                  daq_rate: Union[int, float], counter: Any, reverse=False) -> None:
        """vary one gate that is output of Keithley and record with
        Args:
            gate_grids: Dict of {axis_name: axis_meshgrid} from utils.make_gate_grids().
            ao_channels: Dict of {axis_name: ao_index} for the scanner ao channels.
            daq_rate: DAQ sampling rate in Hz.
            counter: utils.Counter instance, determines current line of the grid.
            reverse: Determines scan direction (i.e. forward or backward).
        """
        daq_name = self.metadata['daq']['name']
        self.ao_task = nidaqmx.Task('DAQ_line_gate_ao_task')
        line = counter.count
        if reverse:
            step = -1
            last_point = 0
        else:
            step = 1
            last_point = -1
        for axis, idx in ao_channels.items():
            if axis == "G":
                if gate_grids['DAQ'] == 'T':
                    out = gate_grids['T'][line][::step]
                    self.ao_task.ao_channels.add_ao_voltage_chan('{}/ao{}'.format(daq_name, idx), 'T')
                elif gate_grids['DAQ'] == 'B':
                    out = gate_grids['B'][line][::step]
                    self.ao_task.ao_channels.add_ao_voltage_chan('{}/ao{}'.format(daq_name, idx), 'B')
        self.ao_task.timing.cfg_samp_clk_timing(daq_rate,
                                                sample_mode=AcquisitionType.FINITE,
                                                samps_per_chan=len(out))

        log.debug('Writing line {}.'.format(line))
        self.ao_task.write(np.array(out), auto_start=False)

    def DAQ_line_gate(self, gate_grids: Dict[str, np.ndarray], ao_channels: Dict[str, int],
                  daq_rate: Union[int, float], counter: Any, reverse=False) -> None:
        """vary one gate that is output of DAC
        Args:
            gate_grids: Dict of {axis_name: axis_meshgrid} from utils.make_gate_grids().
            ao_channels: Dict of {axis_name: ao_index} for the scanner ao channels.
            daq_rate: DAQ sampling rate in Hz.
            counter: utils.Counter instance, determines current line of the grid.
            reverse: Determines scan direction (i.e. forward or backward).
        """
        daq_name = self.metadata['daq']['name']
        self.ao_task = nidaqmx.Task('DAQ_line_gate_ao_task')
        line = counter.count
        if reverse:
            step = -1
            last_point = 0
        else:
            step = 1
            last_point = -1
        for axis, idx in ao_channels.items():
            if axis == "G":
                if gate_grids['DAQ'] == 'T':
                    out = gate_grids['T'][line][::step]
                    self.ao_task.ao_channels.add_ao_voltage_chan('{}/ao{}'.format(daq_name, idx), 'T')
                elif gate_grids['DAQ'] == 'B':
                    out = gate_grids['B'][line][::step]
                    self.ao_task.ao_channels.add_ao_voltage_chan('{}/ao{}'.format(daq_name, idx), 'B')
        self.ao_task.timing.cfg_samp_clk_timing(daq_rate,
                                                sample_mode=AcquisitionType.FINITE,
                                                samps_per_chan=len(out))

        log.debug('Writing line {}.'.format(line))
        self.ao_task.write(np.array(out), auto_start=False)

    def gate_line(self, gate_grids: Dict[str, np.ndarray], ao_channels: Dict[str, int],
                  daq_rate: Union[int, float], counter: Any, reverse=False) -> None:
        """vary one gate that is output of DAC. The gate is assumed as top gate
        Args:
            gate_grids: Dict of {axis_name: axis_meshgrid} from utils.make_gate_grids().
            ao_channels: Dict of {axis_name: ao_index} for the scanner ao channels.
            daq_rate: DAQ sampling rate in Hz.
            counter: utils.Counter instance, determines current line of the grid.
            reverse: Determines scan direction (i.e. forward or backward).
        """
        daq_name = self.metadata['daq']['name']
        self.ao_task = nidaqmx.Task('gate_line_ao_task')
        out = []
        line = counter.count
        if reverse:
            step = -1
            last_point = 0
        else:
            step = 1
            last_point = -1
        for axis, idx in ao_channels.items():
            out.append(gate_grids[axis][line][::step])
            self.ao_task.ao_channels.add_ao_voltage_chan('{}/ao{}'.format(daq_name, idx), axis)
        self.ao_task.timing.cfg_samp_clk_timing(daq_rate,
                                                sample_mode=AcquisitionType.FINITE,
                                                samps_per_chan=len(out[0]))
        log.debug('Writing line {}.'.format(line))
        self.ao_task.write(np.array(out), auto_start=False)
        
    def apply_next_gate(self, gate_grids: Dict[str, np.ndarray], counter: Any, wait: Optional[bool]=False, retractfirst: Optional[bool]=False) -> None:
        """Moves scanner to the start of the next line to scan.
        Args:
            scan_grids: Dict of {axis_name: axis_meshgrid} from utils.make_scan_grids().
            counter: utils.Counter instance, determines current line of the grid.
            wait: wait at the fisrt position of next line before start scan
            retractfirst: retract first when scan finishes
        """
        line = counter.count
        try:
            start_of_next_line = [scan_grids[axis][line+1][0] for axis in ['x', 'y', 'z']]
            self.goto(start_of_next_line, retract_first=retractfirst, quiet=True)
            #wait 5 sec
            if wait:
                time.sleep(5)
        #: If `line` is the last line in the scan, do nothing.
        except IndexError:
            pass

   
   
    def get_gate(self, gate_plot: Any, data_set: Any, counter: Any) -> None:
        """get resistance verses gate voltage in real time plot

        Args:
            gate_plot: plots.gatePlot instance, which contains current data and parameters
                of the gate range.
            data_set: DataSet containing Four_prob data generated by Loop.
            counter: utils.Counter intance to keep track of which point in the Loop we're at.
        """
        #: Resistance in Ohm
        self.gate_value = None
        gate_plot.update(data_set)
        pt = counter.count
        #: If we've reached the last temperature
        if pt >= len(gate_plot.gate):
            return
        #R_unit = RvT_plot['constants']['R_unit']
        #prefactor = self.Q_(RvT_plot.prefactors['Four_prob'])
        #R_lead = self.Q_(RvT_plot.R_lead)
        Fbdata = gate_plot.Fbdata
        Tempdata = gate_plot.gate
        return



    def get_MSvT(self, MSvT_plot: Any, data_set: Any, counter: Any) -> None:
        """get resistance verses temperature in real time plot

        Args:
            MSvT_plot: plots.MSvTPlot instance, which contains current data and parameters
                of the temperature change Loop.
            data_set: DataSet containing MAG and SUSCX data generated by Loop.
            counter: utils.Counter intance to keep track of which point in the Loop we're at.
        """

        MSvT_plot.update(data_set)
        pt = counter.count
        #: If we've reached the last temperature
        if pt >= len(MSvT_plot.Temp):
            return



    def clear_instances(self):
        """Clear scanner instances.
        """
        for inst in self.instances():
            self.remove_instance(inst)
            
    def control_ao_task(self, cmd: str) -> None:
        """Write commands to the DAQ AO Task. Used during qc.Loops.
        Args:
            cmd: What you want the Task to do. For example,
                self.control_ao_task('stop') is equivalent to self.ao_task.stop()
        """
        if hasattr(self, 'ao_task'):
            getattr(self.ao_task, cmd)()

    def make_ramp(self, pos0: List, pos1: List, speed: Union[int, float]) -> np.ndarray:
        """Generates a ramp in x,y,z scanner voltage from point pos0 to point pos1 at given speed.
        Args:
            pos0: List of initial [x, y, z] scanner voltages.
            pos1: List of final [x, y, z] scanner votlages.
            speed: Speed at which to go to pos0 to pos1, in DAQ voltage/second.
        Returns:
            numpy.ndarray: ramp
                Array of x, y, z values to write to DAQ AOs to move
                scanner from pos0 to pos1.
        """
        if speed > self.speed.to('V/s').magnitude:
            msg = 'Setting ramp speed to maximum allowed: {} V/s.'
            log.warning(msg.format(self.speed.to('V/s').magnitude))
        pos0 = np.array(pos0)
        pos1 = np.array(pos1)
        max_ramp_distance = np.max(np.abs(pos1-pos0))
        ramp_time = max_ramp_distance/speed
        npts = int(ramp_time * self.daq_rate) + 2
        ramp = []
        for i in range(3):
            ramp.append(np.linspace(pos0[i], pos1[i], npts))
        return np.array(ramp)

    def gate_ramp(self, old_gate: float, new_gate: float, gate_speed: Union[int, float]) -> np.ndarray:
        """Generates a ramp to change gate voltage.
        Args:
            old_gate: original gate.
            new_gate: gate to ramp to .
            speed: Speed at which gate ramps.
        Returns:
            numpy.ndarray: ramp
                Array to write to DAQ AO to change gate from old_gate to new_gate.
        """
        if gate_speed > self.gate_speed.to('V/s').magnitude:
            msg = 'Setting ramp speed to maximum allowed: {} V/s.'
            log.warning(msg.format(self.gate_speed.to('V/s').magnitude))
        ramp_time = np.abs(old_gate - new_gate)/gate_speed
        npts = int(ramp_time * self.daq_rate) + 2
        ramp = np.linspace(old_gate, new_gate, npts)
        return np.array(ramp)


   
    def _goto_T(self, zpos: float) -> None:
        """Go to a given temperature.
        """
        pass
