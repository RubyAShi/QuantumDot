

#: Various Python utilities
from typing import Dict, List, Sequence, Any, Union, Tuple, Optional
import time

#: Qcodes for running measurements and saving data
import qcodes as qc

#: NI DAQ library
import nidaqmx
from nidaqmx.constants import AcquisitionType, FrequencyUnits, Level

#: scanning-squid modules
from instruments.daq import DAQAnalogInputs
from plots import ScanPlot, MeasPlot
from .microscope import Microscope
import utils
import matplotlib.pyplot as plt

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

class SusceptometerMicroscope(Microscope):
    """Scanning SQUID susceptometer microscope class.
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
        super().__init__(config_file, temp, ureg, log_level, log_name, **kwargs)

    def get_prefactors(self, measurement: Dict[str, Any], update: bool=True) -> Dict[str, Any]:
        """For each channel, calculate prefactors to convert DAQ voltage into real units.
        Args:
            measurement: Dict of measurement parameters as defined
                in measurement configuration file.
            update: Whether to query instrument parameters or simply trust the
                latest values (should this even be an option)?
        Returns:
            Dict[str, pint.Quantity]: prefactors
                Dict of {channel_name: prefactor} where prefactor is a pint Quantity.
        """
        mod_width = self.Q_(self.SQUID.metadata['modulation_width'])
        prefactors = {}
        for ch in measurement['channels']:
            prefactor = 1
            if ch == 'MAG':
                prefactor /= mod_width
            elif ch in ['SUSCX', 'SUSCY']:
                r_lead = self.Q_(measurement['channels'][ch]['r_lead'])
                snap = getattr(self, 'SUSC_lockin').snapshot(update=update)['parameters']
                susc_sensitivity = snap['sensitivity']['value']
                amp = snap['amplitude']['value'] * self.ureg(snap['amplitude']['unit'])
                #: The factor of 10 here is because SR830 output gain is 10/sensitivity
                prefactor *=  (r_lead / amp) / (mod_width * 10 / susc_sensitivity)
            elif ch == 'CAP':
                snap = getattr(self, 'CAP_lockin').snapshot(update=update)['parameters']
                cap_sensitivity = snap['sensitivity']['value']
                #: The factor of 10 here is because SR830 output gain is 10/sensitivity
                prefactor /= (self.Q_(self.scanner.metadata['cantilever']['calibration']) * 10 / cap_sensitivity)
            elif ch == 'LIA':
                r_lead = self.Q_(measurement['channels'][ch]['r_lead'])
                prefactor /= r_lead
            elif ch == 'LIA_curr':
                snap = getattr(self, 'SUSC_lockin').snapshot(update=update)['parameters']
                susc_sensitivity = snap['sensitivity']['value']
                #: The factor of 10 here is because SR830 output gain is 10/sensitivity
                prefactor /= 10 / susc_sensitivity
            elif ch == 'IV':
                snap = getattr(self, 'SUSC_lockin').snapshot(update=update)['parameters']
                r_lead = self.Q_(measurement['channels'][ch]['r_lead'])
                curr_amp = self.Q_(measurement['channels'][ch]['curr_amp'])
                log.info('currently ch1 measuring {}'.format(snap['ch1_display']['value']))
                IV_sensitivity = snap['sensitivity']['value']
                #: The factor of 10 here is because SR830 output gain is 10/sensitivity
                prefactor *= IV_sensitivity/10 * curr_amp

            elif ch == 'Sample_T':
                snap = getattr(self,'ls372').snapshot(update=update)['parameters']
                aout = snap['analog_output']['value']
                aolist = aout.split(",")
                #For Lakeshore372, 0V:Tmin [K], 10V:Tmax [K]
                #return voltage*(Tmax-Tmin)/10 + Tmin
                # Tmin should be zero
                prefactor = self.Q_((float(aolist[4])-float(aolist[5]))/10,'K/V')
            elif ch == 'IVY':
                snap = getattr(self, 'SUSC_lockin').snapshot(update=update)['parameters']
                r_lead = self.Q_(measurement['channels'][ch]['r_lead'])
                amp = snap['amplitude']['value'] * self.ureg(snap['amplitude']['unit'])
                IV_sensitivity = snap['sensitivity']['value']
                if snap['ch2_display']['value'] == "Y":
                    log.info('currently ch2 measuring {}'.format(snap['ch2_display']['value']))
                    #: The factor of 10 here is because SR830 output gain is 10/sensitivity
                    #prefactor *= IV_sensitivity/10 * (r_lead / amp)
                    prefactor *= IV_sensitivity/10 * curr_amp
                    if measurement['channels'][ch]['attenuator'] == '-20dB':
                        prefactor *= 10
                elif snap['ch2_display']['value'] == "Phase":
                    # phase is 18 degree per volt
                    # e.g. 180deg is 10V
                    log.info('currently ch2 measuring {}'.format(snap['ch2_display']['value']))
                    prefactor *= 18 * self.ureg(snap['phase']['unit']) / self.ureg('V') 
            prefactor = self.ureg.Quantity(str(prefactor))
            prefactor /= measurement['channels'][ch]['gain']
            prefactors.update({ch: prefactor.to('{}/V'.format(measurement['channels'][ch]['unit']))})
        return prefactors

    
    def plot_T_vs_time(self):
        time_vec=[]
        sample_T=[]
        elapsed_time=0
        self.fig, self.ax = plt.subplots(1,1)
        self.ax.set_xlabel('t(sec)')
        self.ax.set_ylabel('T(K)')
         
        while (True):
            sample_T.append(self.ls340.A.temperature())
            time_vec.append(elapsed_time)
            time.sleep(1)
            elapsed_time=elapsed_time+1
            self.ax.plot(time_vec,sample_T)
            self.fig.canvas.draw()


 
