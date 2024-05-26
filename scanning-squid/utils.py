import os
import numpy as np
from typing import Dict, List, Optional, Sequence, Any, Union, Tuple, Callable
import qcodes as qc
from qcodes.instrument.parameter import ArrayParameter, Parameter
from qcodes.instrument.base import Instrument
import qcodes.utils.validators as vals
from scipy import io
from collections import OrderedDict
import json

#: Tell the UnitRegistry what a Phi0 is, and that ohm and Ohm are the same thing.
with open('squid_units.txt', 'w') as f:
    f.write('Phi0 = 2.067833831e-15 * Wb\n')
    f.write('Ohm = ohm\n')


class Counter(object):
    """Simple counter used to keep track of progress in a Loop.
    """
    def __init__(self):
        self.count = 0
        
    def advance(self):
        self.count += 1

def load_json_ordered(filename: str) -> OrderedDict:
    """Loads json file as an ordered dict.
    Args:
        filename: Path to json file to be loaded.
    Returns:
        OrderedDict: odict
            OrderedDict containing data from json file.
    """
    with open(filename) as f:
        odict = json.load(f, object_pairs_hook=OrderedDict)
    return odict
        
def next_file_name(fpath: str, extension: str) -> str:
    """Appends an integer to fpath to create a unique file name:
        fpath + {next unused integer} + '.' + extension
    Args:
        fpath: Path to file you want to create (no extension).
        extension: Extension of file you want to create.
    Returns:
        str: next_file_name
            Unique file name starting with fpath and ending with extension.
    """
    i = 0
    while os.path.exists('{}{}.{}'.format(fpath, i, extension)):
        i += 1
    return '{}{}.{}'.format(fpath, i, extension)

def make_gate_vectors(gate_params: Dict[str, Any], ureg: Any) -> Dict[str, Sequence[float]]:
    """Creates x and y vectors for given scan parameters.
    Args:
        scan_params: Scan parameter dict
        ureg: pint UnitRegistry, manages units.
    Returns:
        Dict: scan_vectors
            {axis_name: axis_vector} for x, y axes.
    """
    Q_ = ureg.Quantity
    for ax in ['X', 'Y']:
        if ax == 'Y':
            Ymin = Q_(gate_params['VYrange'][0]).to('V').magnitude
            Ymax = Q_(gate_params['VYrange'][1]).to('V').magnitude
            Ysize = Q_(gate_params['dVY']).to('V').magnitude
        if ax == 'X':
            Xmin = Q_(gate_params['VXrange'][0]).to('V').magnitude
            Xmax = Q_(gate_params['VXrange'][1]).to('V').magnitude
            Xsize = Q_(gate_params['dVX']).to('V').magnitude
    YG = np.arange(Ymin, Ymax + Ysize * 1/10000, Ysize)
    XG = np.arange(Xmin, Xmax + Xsize * 1/10000, Xsize)
    return {'Y': YG, 'X': XG, "fast_ax": gate_params["fast_ax"]}

def make_gate_grids(gate_vectors: Dict[str, Sequence[float]], fast_ax_pts: int
                    ) -> Dict[str, Any]:
    """Makes meshgrids of top and bottom gate to write to DAQ analog outputs.
    Args:
        gate_vectors: Dict of {axis_name: axis_vector} for T(y), B(x) axes (from make_gate_vectors).
        slow_ax: Name of the scan slow axis ('T' or 'B').
        fast_ax: Name of the scan fast axis ('T' or 'B').
        fast_ax_pts: Number of points to write to DAQ analog outputs to scan fast axis.
    Returns:
        Dict: scan_grids
            {axis_name: axis_scan_grid} for x, y, z, axes.
    """
    fast_ax = gate_vectors["fast_ax"]
    slow_ax = 'X' if fast_ax == 'Y' else 'Y'
    slow_ax_vec = gate_vectors[slow_ax]
    fast_ax_vec = np.linspace(gate_vectors[fast_ax][0],
                              gate_vectors[fast_ax][-1],
                              fast_ax_pts)
    if fast_ax == 'Y':
        X, Y = np.meshgrid(slow_ax_vec, fast_ax_vec, indexing='ij')
    else:
        X, Y = np.meshgrid(fast_ax_vec, slow_ax_vec, indexing='xy')
    return {'Y': Y, 'X': X, 'fast_ax': fast_ax}




def to_real_units(data_set: Any, ureg: Any=None) -> Any:
    """Converts DataSet arrays from DAQ voltage to real units using recorded metadata.
        Preserves shape of DataSet arrays.
    Args:
        data_set: qcodes DataSet created by Microscope.scan_plane
        ureg: Pint UnitRegistry. Default None.
        
    Returns:
        np.ndarray: data
            ndarray like the DataSet array, but in real units as prescribed by
            factors in DataSet metadata.
    """
    if ureg is None:
        from pint import UnitRegistry
        ureg = UnitRegistry()
        ureg.load_definitions('./squid_units.txt')
    meta = data_set.metadata['loop']['metadata']
    data = np.full_like(data_set.daq_ai_voltage, np.nan, dtype=np.double)
    for i, ch in enumerate(meta['channels'].keys()):
        array = data_set.daq_ai_voltage[:,i,:] * ureg('V')
        unit = meta['channels'][ch]['unit']
        data[:,i,:] = (array * ureg.Quantity(meta['prefactors'][ch])).to(unit)
    return data

def to_real_units_double_loops(data_set: Any, ureg: Any=None) -> Any:
    """Converts DataSet arrays from DAQ voltage to real units using recorded metadata.
        Preserves shape of DataSet arrays.
    Args:
        data_set: qcodes DataSet created by Microscope.scan_plane
        ureg: Pint UnitRegistry. Default None.
        
    Returns:
        np.ndarray: data
            ndarray like the DataSet array, but in real units as prescribed by
            factors in DataSet metadata.
    """
    if ureg is None:
        from pint import UnitRegistry
        ureg = UnitRegistry()
        ureg.load_definitions('./squid_units.txt')
    meta = data_set.metadata['loop']['metadata']
    data = np.full_like(data_set.daq_ai_voltage, np.nan, dtype=np.double)
    #print(data.shape)
    #print(data)
    for i, ch in enumerate(meta['channels'].keys()):
        array = data_set.daq_ai_voltage[:,:,i,:] * ureg('V')
        unit = meta['channels'][ch]['unit']
        data[:,:,i,:] = (array * ureg.Quantity(meta['prefactors'][ch])).to(unit)
    return data

def double_gate_to_arrays(gate_data: Any, ureg: Optional[Any]=None, real_units: Optional[bool]=True,
                   xy_unit: Optional[str]=None) -> Dict[str, Any]:
    """Extracts scan data from DataSet and converts to requested units.
    Args:
        gate_data: qcodes DataSet created by Microscope.scan_plane
        ureg: pint UnitRegistry, manages physical units.
        real_units: If True, converts z-axis data from DAQ voltage into
            units specified in measurement configuration file.
        xy_unit: String describing quantity with dimensions of length.
            If xy_unit is not None, scanner x, y DAQ ao voltage will be converted to xy_unit
            according to scanner constants defined in microscope configuration file.
    Returns:
        Dict: arrays
            Dict of x, y vectors and grids, and measured data in requested units.
    """
    if ureg is None:
        from pint import UnitRegistry
        ureg = UnitRegistry()
        #: Tell the UnitRegistry what a Phi0 is, and that ohm and Ohm are the same thing.
        with open('squid_units.txt', 'w') as f:
            f.write('Phi0 = 2.067833831e-15 * Wb\n')
            f.write('Ohm = ohm\n')
        ureg.load_definitions('./squid_units.txt')
    Q_ = ureg.Quantity
    meta = gate_data.metadata['loop']['metadata']
    gate_vectors = make_gate_vectors(meta, ureg)
    fast_ax = meta['fast_ax'] 
    slow_ax = 'Y' if fast_ax == 'X' else 'X'
    num = len(gate_vectors[fast_ax])
    grids = make_gate_grids(gate_vectors, num)
    arrays = {'X_grid': grids['X'] * ureg('V'), 'Y_grid': grids['Y']* ureg('V')}
    arrays.update({'X': gate_vectors['X'] * ureg('V'), 'Y': gate_vectors['Y'] * ureg('V')})
    for ch, info in meta['channels'].items():
        array = gate_data.daq_ai_voltage[:,:,info['idx'],:] * ureg('V')
        if real_units:
            pre = meta['prefactors'][ch]
            arrays.update({ch: (Q_(pre) * array).to(info['unit'])})
        else:
            arrays.update({ch: array})
    if real_units and xy_unit is not None:
        bendc = scan_data.metadata['station']['instruments']['benders']['metadata']['constants']
        for ax in ['X', 'Y']:
            grid = (grids[ax] * ureg('V') * Q_(bendc[ax])).to(xy_unit)
            vector = (scan_vectors[ax] * ureg('V') * Q_(bendc[ax])).to(xy_unit)
            arrays.update({ax.upper(): grid, ax: vector})
    return arrays

def double_gate_to_arrays_old(gate_data: Any, ureg: Optional[Any]=None, real_units: Optional[bool]=True,
                   xy_unit: Optional[str]=None) -> Dict[str, Any]:
    """Extracts scan data from DataSet and converts to requested units.
    Args:
        gate_data: qcodes DataSet created by Microscope.scan_plane
        ureg: pint UnitRegistry, manages physical units.
        real_units: If True, converts z-axis data from DAQ voltage into
            units specified in measurement configuration file.
        xy_unit: String describing quantity with dimensions of length.
            If xy_unit is not None, scanner x, y DAQ ao voltage will be converted to xy_unit
            according to scanner constants defined in microscope configuration file.
    Returns:
        Dict: arrays
            Dict of x, y vectors and grids, and measured data in requested units.
    """
    if ureg is None:
        from pint import UnitRegistry
        ureg = UnitRegistry()
        #: Tell the UnitRegistry what a Phi0 is, and that ohm and Ohm are the same thing.
        with open('squid_units.txt', 'w') as f:
            f.write('Phi0 = 2.067833831e-15 * Wb\n')
            f.write('Ohm = ohm\n')
        ureg.load_definitions('./squid_units.txt')
    Q_ = ureg.Quantity
    meta = gate_data.metadata['loop']['metadata']
    gate_vectors = make_gate_vectors(meta, ureg)
    fast_ax = meta['DAQ_AO'] 
    slow_ax = 'T' if fast_ax == 'B' else 'B'
    num = len(gate_vectors[fast_ax])
    grids = make_gate_grids(gate_vectors, slow_ax, fast_ax, num)
    arrays = {'B_grid': grids['B'] * ureg('V'), 'T_grid': grids['T']* ureg('V')}
    arrays.update({'B': gate_vectors['B'] * ureg('V'), 'T': gate_vectors['T'] * ureg('V')})
    for ch, info in meta['channels'].items():
        array = gate_data.daq_ai_voltage[:,info['idx'],:] * ureg('V')
        if real_units:
            pre = meta['prefactors'][ch]
            arrays.update({ch: (Q_(pre) * array).to(info['unit'])})
        else:
            arrays.update({ch: array})
    if real_units and xy_unit is not None:
        bendc = scan_data.metadata['station']['instruments']['benders']['metadata']['constants']
        for ax in ['B', 'T']:
            grid = (grids[ax] * ureg('V') * Q_(bendc[ax])).to(xy_unit)
            vector = (scan_vectors[ax] * ureg('V') * Q_(bendc[ax])).to(xy_unit)
            arrays.update({ax.upper(): grid, ax: vector})
    return arrays


def gate_to_arrays(gate_data: Any, ureg: Optional[Any]=None, real_units: Optional[bool]=True) -> Dict[str, Any]:
    """Extracts gated IV data from DataSet and converts to requested units.

    Args:
        gate_data: qcodes DataSet created by Microscope.td_cap
        ureg: pint UnitRegistry, manages physical units.
        real_units: If True, converts data from DAQ voltage into
            units specified in measurement configuration file.
    Returns:
        Dict: arrays
            Dict of measured data in requested units.
    """
    if ureg is None:
        from pint import UnitRegistry
        ureg = UnitRegistry()
        #: Tell the UnitRegistry what a Phi0 is, and that ohm and Ohm are the same thing.
        with open('squid_units.txt', 'w') as f:
            f.write('Phi0 = 2.067833831e-15 * Wb\n')
            f.write('Ohm = ohm\n')
        ureg.load_definitions('./squid_units.txt')
    Q_ = ureg.Quantity
    meta = gate_data.metadata['loop']['metadata']
    G = [Q_(val).to('V').magnitude for val in meta['range']]
    direction = meta['direction']
    dV = Q_(meta['dV']).to('V').magnitude
    if direction == 'backwards':
        G_array = np.linspace(G[1], G[0], int((G[1]-G[0])/dV) + 1)
    else:
        G_array = np.linspace(G[0], G[1], int((G[1]-G[0])/dV) + 1)
    arrays = {'gate': G_array * ureg('V')}
    for ch, info in meta['channels'].items():
        array = gate_data.daq_ai_voltage[:,info['idx'],0] * ureg('V')
        if real_units:
            pre = meta['prefactors'][ch]
            arrays.update({ch: (Q_(pre) * array).to(info['unit'])})
        else:
            arrays.update({ch: array})
    return arrays

def meas_to_arrays(meas_data: Any, co_freq:Union[int,float],
                   scans_per_period:int, ureg: Optional[Any]=None, real_units: Optional[bool]=True) -> Dict[str, Any]:
    """Extracts ring measurement data from DataSet and converts to requested units.
    Args:
        meas_data: qcodes DataSet created by Microscope.meas_ring
        ureg: pint UnitRegistry, manages physical units.
        real_units: If True, converts z-axis data from DAQ voltage into
            units specified in measurement configuration file.
    Returns:
        Dict: arrays
            Dict of block number measured data in requested units.
    """
    if ureg is None:
        from pint import UnitRegistry
        ureg = UnitRegistry()
        #: Tell the UnitRegistry what a Phi0 is, and that ohm and Ohm are the same thing.
        with open('squid_units.txt', 'w') as f:
            f.write('Phi0 = 2.067833831e-15 * Wb\n')
            f.write('Ohm = ohm\n')
        ureg.load_definitions('./squid_units.txt')
    Q_ = ureg.Quantity
    meta = meas_data.metadata['loop']['metadata']
    time = np.linspace(0, 1 / co_freq, scans_per_period)
    #: Make the nominal FC output axis
    r_lead = Q_(meta['channels']['LIA']['r_lead']).to('Ohm').magnitude
    LIA_amp = Q_(meta['channels']['LIA']['lockin']['amplitude']).to('V').magnitude
    ifc_full = LIA_amp / r_lead * np.sqrt(2) * np.sin(2 * np.pi * time * co_freq)
    ifc = np.mean(np.reshape(ifc_full, newshape=(scans_per_period, -1)), axis=1)
    arrays = {'nominal_ifc': ifc * ureg('A')}
    for ch, info in meta['channels'].items():
        array = meas_data.daq_ai_voltage[:,info['idx'],:] * ureg('V')
        if real_units:
            pre = meta['prefactors'][ch]
            arrays.update({ch: (Q_(pre) * array).to(info['unit'])})
        else:
            arrays.update({ch: array})
    return arrays


def MSvT_to_arrays(MS_data: Any, ureg: Optional[Any]=None, real_units: Optional[bool]=True) -> Dict[str, Any]:
    """Extracts scan data from DataSet and converts to requested units.

    Args:
        MS_data: qcodes DataSet created by Microscope.temp_series
        ureg: pint UnitRegistry, manages physical units.
        real_units: If True, converts data from DAQ voltage into
            units specified in measurement configuration file.
    Returns:
        Dict: arrays
            Dict of measured data in requested units.
    """
    if ureg is None:
        from pint import UnitRegistry
        ureg = UnitRegistry()
        #: Tell the UnitRegistry what a Phi0 is, and that ohm and Ohm are the same thing.
        with open('squid_units.txt', 'w') as f:
            f.write('Phi0 = 2.067833831e-15 * Wb\n')
            f.write('Ohm = ohm\n')
        ureg.load_definitions('./squid_units.txt')
    Q_ = ureg.Quantity
    meta = MS_data.metadata['loop']['metadata']
    T = [Q_(val).to('K').magnitude for val in meta['range']]
    
    dT = Q_(meta['dT']).to('K').magnitude
    T_array = np.linspace(T[0], T[1], int((T[1]-T[0])/dT) + 1)
    arrays = {'temp': T_array * ureg('K')}
    for ch, info in meta['channels'].items():
        array = MS_data.daq_ai_voltage[:,info['idx'],0] * ureg('V')
        if real_units:
            pre = meta['prefactors'][ch]
            arrays.update({ch: (Q_(pre) * array).to(info['unit'])})
        else:
            arrays.update({ch: array})
    return arrays


def double_gate_to_mat_file(gate_data: Any, real_units: Optional[bool]=True, xy_unit: Optional[bool]=None,
    fname: Optional[str]=None, interpolator: Optional[Callable]=None) -> None:
    """Export double gate map.
    Args:
        gate_data: qcodes DataSet created by Microscope.scan_plane
        real_units: If True, converts z-axis data from DAQ voltage into
            units specified in measurement configuration file.
        xy_unit: String describing quantity with dimensions of length.
            If xy_unit is not None, scanner x, y DAQ ao voltage will be converted to xy_unit
            according to scanner constants defined in microscope configuration file.
        fname: File name (without extension) for resulting .mat file.
            If None, uses the file name defined in measurement configuration file.
        interpolator: Instance of scipy.interpolate.Rbf, used to interpolate touchdown points.
            Default: None.
    """
    from pint import UnitRegistry
    ureg = UnitRegistry()
    ureg.load_definitions('./squid_units.txt')
    Q_ = ureg.Quantity
    meta = gate_data.metadata['loop']['metadata']
    arrays = double_gate_to_arrays(gate_data, ureg=ureg, real_units=real_units, xy_unit=xy_unit)
    mdict = {}
    for name, arr in arrays.items():
        if real_units:
            if xy_unit:
                unit = meta['channels'][name]['unit'] if name not in ['X', 'Y', 'X_grid', 'Y_grid'] else xy_unit
            else: 
                unit = meta['channels'][name]['unit'] if name not in ['X', 'Y', 'X_grid', 'Y_grid'] else 'V'
        else:
            unit = 'V'
        if meta['fast_ax'] == 'Y':
            arr = arr.T
        mdict.update({name: {'array': arr.to(unit).magnitude, 'unit': unit}})
    mdict.update({'prefactors': meta['prefactors'], 'location': gate_data.location})
    if fname is None:
        fname = meta['fname']
    fpath = gate_data.location + '/'
    io.savemat(next_file_name(fpath + fname, 'mat'), mdict)


def double_gate_to_mat_file_old(gate_data: Any, real_units: Optional[bool]=True, xy_unit: Optional[bool]=None,
    fname: Optional[str]=None, interpolator: Optional[Callable]=None) -> None:
    """Export double gate map.
    Args:
        gate_data: qcodes DataSet created by Microscope.scan_plane
        real_units: If True, converts z-axis data from DAQ voltage into
            units specified in measurement configuration file.
        xy_unit: String describing quantity with dimensions of length.
            If xy_unit is not None, scanner x, y DAQ ao voltage will be converted to xy_unit
            according to scanner constants defined in microscope configuration file.
        fname: File name (without extension) for resulting .mat file.
            If None, uses the file name defined in measurement configuration file.
        interpolator: Instance of scipy.interpolate.Rbf, used to interpolate touchdown points.
            Default: None.
    """
    from pint import UnitRegistry
    ureg = UnitRegistry()
    ureg.load_definitions('./squid_units.txt')
    Q_ = ureg.Quantity
    meta = gate_data.metadata['loop']['metadata']
    arrays = double_gate_to_arrays(gate_data, ureg=ureg, real_units=real_units, xy_unit=xy_unit)
    mdict = {}
    for name, arr in arrays.items():
        if real_units:
            if xy_unit:
                unit = meta['channels'][name]['unit'] if name not in ['T', 'B', 'T_grid', 'B_grid'] else xy_unit
            else: 
                unit = meta['channels'][name]['unit'] if name not in ['T', 'B', 'T_grid', 'B_grid'] else 'V'
        else:
            unit = 'V'
        if meta['DAQ_AO'] == 'T':
            arr = arr.T
        mdict.update({name: {'array': arr.to(unit).magnitude, 'unit': unit}})
    mdict.update({'prefactors': meta['prefactors'], 'location': gate_data.location})
    if fname is None:
        fname = meta['fname']
    fpath = gate_data.location + '/'
    io.savemat(next_file_name(fpath + fname, 'mat'), mdict)


def gate_to_mat_file(gate_data: Any, real_units: Optional[bool]=True, fname: Optional[str]=None) -> None:
    """Export DataSet created by microscope.td_cap to .mat file for analysis.

    Args:
        R_data: qcodes DataSet created by Microscope.Four_prob
        real_units: If True, converts data from DAQ voltage into
            units specified in measurement configuration file.
        fname: File name (without extension) for resulting .mat file.
            If None, uses the file name defined in measurement configuration file.
    """
    from pint import UnitRegistry
    ureg = UnitRegistry()
    ureg.load_definitions('./squid_units.txt')
    Q_ = ureg.Quantity
    meta = gate_data.metadata['loop']['metadata']
    arrays = gate_to_arrays(gate_data, ureg=ureg, real_units=real_units)
    mdict = {}
    for name, arr in arrays.items():
        if name is not 'gate':
            unit = meta['channels'][name]['unit'] if real_units else 'V'
            mdict.update({name: {'array': arr.to(unit).magnitude, 'unit': unit}})
    mdict.update({'gate': {'array': arrays['gate'].to('V').magnitude, 'unit': 'V'}})
    mdict.update({
        'prefactors': meta['prefactors'],
        })
    if fname is None:
        fname = 'gated_IV'
    fpath = gate_data.location + '/'
    io.savemat(next_file_name(fpath + fname, 'mat'), mdict)



def MSvT_to_mat_file(MS_data: Any, real_units: Optional[bool]=True, fname: Optional[str]=None) -> None:
    """Export DataSet created by microscope.td_cap to .mat file for analysis.

    Args:
        MS_data: qcodes DataSet created by Microscope.temp_series
        real_units: If True, converts data from DAQ voltage into
            units specified in measurement configuration file.
        fname: File name (without extension) for resulting .mat file.
            If None, uses the file name defined in measurement configuration file.
    """
    from pint import UnitRegistry
    ureg = UnitRegistry()
    ureg.load_definitions('./squid_units.txt')
    Q_ = ureg.Quantity
    meta = MS_data.metadata['loop']['metadata']
    arrays = MSvT_to_arrays(MS_data, ureg=ureg, real_units=real_units)
    mdict = {}
    for name, arr in arrays.items():
        if name is not 'temp':
            unit = meta['channels'][name]['unit'] if real_units else 'V'
            mdict.update({name: {'array': arr.to(unit).magnitude, 'unit': unit}})
    mdict.update({'temp': {'array': arrays['temp'].to('K').magnitude, 'unit': 'K'}})
    mdict.update({
        'prefactors': meta['prefactors'],
        })
    if fname is None:
        fname = 'MSvT'
    fpath = MS_data.location + '/'
    io.savemat(next_file_name(fpath + fname, 'mat'), mdict)
    
def clear_artists(ax):
    for artist in ax.lines + ax.collections:
        artist.remove()
