from typing import List, Dict, Any
from .vehicle import Vehicle
from .rsu import RSU
from .uav import UAV
from .irs import IRS, IRSElement
from .task import Task

class QFRLEnvironment:
    """
    Basic Simulation environment holding the physical entities described in the paper.
    """
    def __init__(self):
        self.vehicles: List[Vehicle] = []
        self.rsus: List[RSU] = []
        self.uav: UAV = None
        self.irs: IRS = None
        self.tasks: List[Task] = []

    def initialize_from_config(self, config: Dict[str, Any]):
        """
        Initializes the environment entities from the provided configuration dictionary.
        Uses default positions as mobility is not yet implemented.
        """
        # Vehicles
        veh_cfg = config.get('vehicles', {})
        count = veh_cfg.get('count', 0)
        for i in range(count):
            self.vehicles.append(Vehicle(
                vehicle_id=i,
                x=0.0,
                y=0.0,
                speed=veh_cfg.get('max_velocity_kmh', 40.0),
                cpu_frequency=veh_cfg.get('cpu_frequency_ghz', 4.0),
                max_transmit_power=veh_cfg.get('max_transmit_power_dbm', 20.0)
            ))
            
        # RSU (Configurable, defaulting to 1 for basic initialization if present)
        rsu_cfg = config.get('rsu', {})
        if rsu_cfg:
            self.rsus.append(RSU(
                rsu_id=0,
                x=0.0,
                y=0.0,
                cpu_frequency=rsu_cfg.get('cpu_frequency_ghz', 9.0)
            ))

        # UAV
        uav_cfg = config.get('uav', {})
        if uav_cfg:
            self.uav = UAV(
                x=0.0,
                y=0.0,
                altitude=uav_cfg.get('altitude_m', 300.0),
                max_speed=uav_cfg.get('max_velocity_kmh', 36.0),
                cpu_frequency=uav_cfg.get('cpu_frequency_ghz', 12.0)
            )

        # IRS
        irs_cfg = config.get('irs', {})
        num_elements = irs_cfg.get('num_elements', 0) if irs_cfg else 0
        elements = [IRSElement(i, 0.0, 0.0, 0.0) for i in range(num_elements)]
        self.irs = IRS(
            num_elements=num_elements,
            base_x=0.0,
            base_y=0.0,
            base_z=0.0,
            elements=elements
        )
