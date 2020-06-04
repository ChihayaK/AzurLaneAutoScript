import numpy as np

from module.base.utils import location2node, node2location
from module.logger import logger
from module.map.grid_info import GridInfo
from module.map.map_grids import SelectedGrids


def location_ensure(location):
    if isinstance(location, GridInfo):
        return location.location
    elif isinstance(location, str):
        return node2location(location)
    else:
        return location


def camera_1d(shape, sight):
    start, step = abs(sight[0]), sight[1] - sight[0] + 1
    if shape <= start:
        out = shape // 2
    else:
        out = list(range(start, 26, step))
        out.append(shape - sight[1])
        out = [x for x in set(out) if x <= shape - sight[1]]
    return out


def camera_2d(shape, sight):
    x = camera_1d(shape=shape[0], sight=[sight[0], sight[2]])
    y = camera_1d(shape=shape[1], sight=[sight[1], sight[3]])
    out = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    return [tuple(c) for c in out]


class CampaignMap:
    def __init__(self, name=None):
        self.name = name
        self.grids = {}
        self._shape = (0, 0)
        self._map_data = ''
        self._weight_data = ''
        self._wall_data = ''
        self._block_data = []
        self._spawn_data = []
        self._spawn_data_backup = []
        self._camera_data = []
        self.in_map_swipe_preset_data = None
        self.poor_map_data = False
        self.camera_sight = (-3, -1, 3, 2)
        self.grid_connection = {}

    def __iter__(self):
        return iter(self.grids.values())

    def __getitem__(self, item):
        """
        Args:
            item:

        Returns:
            GridInfo:
        """
        return self.grids[tuple(item)]

    def __contains__(self, item):
        return tuple(item) in self.grids

    @staticmethod
    def _parse_text(text):
        text = text.strip()
        for y, row in enumerate(text.split('\n')):
            row = row.strip()
            for x, data in enumerate(row.split(' ')):
                yield (x, y), data

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, scale):
        self._shape = node2location(scale.upper())
        for y in range(self._shape[1] + 1):
            for x in range(self._shape[0] + 1):
                grid = GridInfo()
                grid.location = (x, y)
                self.grids[(x, y)] = grid

        # camera_data can be generate automatically, but it's better to set it manually.
        self.camera_data = [location2node(loca) for loca in camera_2d(self._shape, sight=self.camera_sight)]
        # weight_data set to 10.
        for grid in self:
            grid.weight = 10.
        # Initialize grid connection.
        self.grid_connection_initial()

    @property
    def map_data(self):
        return self._map_data

    @map_data.setter
    def map_data(self, text):
        if not len(self.grids.keys()):
            grids = np.array([loca for loca, _ in self._parse_text(text)])
            self.shape = location2node(tuple(np.max(grids, axis=0)))

        self._map_data = text
        for loca, data in self._parse_text(text):
            self.grids[loca].decode(data)

    @property
    def wall_data(self):
        return self._wall_data

    @wall_data.setter
    def wall_data(self, text):
        self._wall_data = text

    def grid_connection_initial(self, wall=False):
        """
        Args:
            wall (bool): If use wall_data

        Returns:
            bool: If used wall data.
        """
        # Generate grid connection.
        total = set([grid for grid in self.grids.keys()])
        for grid in self:
            connection = set()
            for arr in np.array([(0, -1), (0, 1), (-1, 0), (1, 0)]):
                arr = tuple(arr + grid.location)
                if arr in total:
                    connection.add(self[arr])
            self.grid_connection[grid] = connection

        if not wall or not self._wall_data:
            return False

        # Use wall_data to delete connection.
        wall = []
        for y, line in enumerate([l for l in self._wall_data.split('\n') if l]):
            for x, letter in enumerate(line[4:-2]):
                if letter != ' ':
                    wall.append((x, y))
        wall = np.array(wall)
        vert = wall[np.all([wall[:, 0] % 4 == 2, wall[:, 1] % 2 == 0], axis=0)]
        hori = wall[np.all([wall[:, 0] % 4 == 0, wall[:, 1] % 2 == 1], axis=0)]
        disconnect = []
        for loca in (vert - (2, 0)) // (4, 2):
            disconnect.append([loca, loca + (1, 0)])
        for loca in (hori - (0, 1)) // (4, 2):
            disconnect.append([loca, loca + (0, 1)])
        for g1, g2 in disconnect:
            g1 = self[g1]
            g2 = self[g2]
            self.grid_connection[g1].remove(g2)
            self.grid_connection[g2].remove(g1)

        return True

    def show(self):
        # logger.info('Showing grids:')
        logger.info('  ' + ' '.join([' ' + chr(x + 64 + 1) for x in range(self.shape[0] + 1)]))
        for y in range(self.shape[1] + 1):
            text = str(y + 1) + ' ' + ' '.join(
                [self[(x, y)].str if (x, y) in self else '  ' for x in range(self.shape[0] + 1)])
            logger.info(text)

    def update(self, grids, camera, is_carrier_scan=False):
        """
        Args:
            grids:
            camera (tuple):
            is_carrier_scan (bool):
        """
        offset = np.array(camera) - np.array(grids.center_grid)
        grids.show()
        for grid in grids.grids.values():
            loca = tuple(offset + grid.location)
            if loca in self.grids:
                self.grids[loca].update(grid, is_carrier_scan=is_carrier_scan, ignore_may=self.poor_map_data)

        return True

    def reset(self):
        for grid in self:
            grid.reset()

    def reset_fleet(self):
        for grid in self:
            grid.is_current_fleet = False

    @property
    def camera_data(self):
        """
        Returns:
            SelectedGrids:
        """
        return self._camera_data

    @camera_data.setter
    def camera_data(self, nodes):
        """
        Args:
            nodes (list): Contains str.
        """
        self._camera_data = SelectedGrids([self[node2location(node)] for node in nodes])

    @property
    def spawn_data(self):
        return self._spawn_data

    @spawn_data.setter
    def spawn_data(self, data_list):
        self._spawn_data_backup = data_list
        spawn = {'battle': 0, 'enemy': 0, 'mystery': 0, 'siren': 0, 'boss': 0}
        for data in data_list:
            spawn['battle'] = data['battle']
            spawn['enemy'] += data.get('enemy', 0)
            spawn['mystery'] += data.get('mystery', 0)
            spawn['siren'] += data.get('siren', 0)
            spawn['boss'] += data.get('boss', 0)
            self._spawn_data.append(spawn.copy())

    @property
    def weight_data(self):
        return self._weight_data

    @weight_data.setter
    def weight_data(self, text):
        self._weight_data = text
        for loca, data in self._parse_text(text):
            self[loca].weight = float(data)

    @property
    def is_map_data_poor(self):
        if not self.select(may_enemy=True) or not self.select(may_boss=True) or not self.select(is_spawn_point=True):
            return False
        if not len(self._spawn_data_backup):
            return False
        return True

    def show_cost(self):
        logger.info('  ' + ' '.join(['   ' + chr(x + 64 + 1) for x in range(self.shape[0] + 1)]))
        for y in range(self.shape[1] + 1):
            text = str(y + 1) + ' ' + ' '.join(
                [str(self[(x, y)].cost).rjust(4) if (x, y) in self else '    ' for x in range(self.shape[0] + 1)])
            logger.info(text)

    def show_connection(self):
        logger.info('  ' + ' '.join([' ' + chr(x + 64 + 1) for x in range(self.shape[0] + 1)]))
        for y in range(self.shape[1] + 1):
            text = str(y + 1) + ' ' + ' '.join(
                [location2node(self[(x, y)].connection) if (x, y) in self and self[(x, y)].connection else '  ' for x in
                 range(self.shape[0] + 1)])
            logger.info(text)

    def find_path_initial(self, location, has_ambush=True):
        location = location_ensure(location)

        ambush_cost = 10 if has_ambush else 1
        for grid in self:
            grid.cost = 9999
            grid.connection = None
        start = self[location]
        start.cost = 0
        visited = [start]
        visited = set(visited)

        while 1:
            new = visited.copy()
            for grid in visited:
                for arr in self.grid_connection[grid]:
                    if arr.is_land:
                        continue
                    cost = 1 if arr.is_ambush_save else ambush_cost
                    cost += grid.cost

                    if cost < arr.cost:
                        arr.cost = cost
                        arr.connection = grid.location
                    elif cost == arr.cost:
                        if abs(arr.location[0] - grid.location[0]) == 1:
                            arr.connection = grid.location
                    if arr.is_sea:
                        new.add(arr)
            if len(new) == len(visited):
                break
            visited = new

        # self.show_cost()
        # self.show_connection()

    def _find_path(self, location):
        """

        Args:
            location (tuple):

        Returns:
            list[tuple]: walking route.

        Examples:
            MAP_7_2._find_path(node2location('H2'))
            [(2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (6, 1), (7, 1)]  # ['C3', 'D3', 'E3', 'F3', 'G3', 'G2', 'H2']
        """
        if self[location].cost == 0:
            return [location]
        if self[location].connection is None:
            return None
        res = [location]
        while 1:
            location = self[location].connection
            if len(res) > 30:
                logger.warning('Route too long')
                logger.warning(res)
                # exit(1)
            if location is not None:
                res.append(location)
            else:
                break
        res.reverse()

        if len(res) == 0:
            logger.warning('No path found. Destination: %s' % str(location))
            return [location, location]

        return res

    def _find_route_node(self, route, step=0):
        """

        Args:
            route (list[tuple]): list of grids.
            step (int): Fleet step in event map. Default to 0.

        Returns:
            list[tuple]: list of walking node.

        Examples:
            MAP_7_2._find_route_node([(2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (6, 1), (7, 1)])
            [(6, 2), (7, 1)]
        """
        res = []
        diff = np.abs(np.diff(route, axis=0))
        turning = np.diff(diff, axis=0)[:, 0]
        indexes = np.where(turning == -1)[0] + 1
        for index in indexes:
            if not self[route[index]].is_fleet:
                res.append(index)
            else:
                logger.info(f'Path_node_avoid: {self[route[index]]}')
                if (index > 1) and (index - 1 not in indexes):
                    res.append(index - 1)
                if (index < len(route) - 2) and (index + 1 not in indexes):
                    res.append(index + 1)
        res.append(len(route) - 1)
        # res = [6, 8]
        if step == 0:
            return [route[index] for index in res]

        res.insert(0, 0)
        inserted = []
        for left, right in zip(res[:-1], res[1:]):
            for index in list(range(left, right, step))[1:]:
                if not self[route[index]].is_fleet:
                    inserted.append(index)
                else:
                    logger.info(f'Path_node_avoid: {self[route[index]]}')
                    if (index > 1) and (index - 1 not in res):
                        inserted.append(index - 1)
                    if (index < len(route) - 2) and (index + 1 not in res):
                        inserted.append(index + 1)
            inserted.append(right)
        res = inserted
        # res = [3, 6, 8]
        return [route[index] for index in res]

    def find_path(self, location, step=0):
        location = location_ensure(location)

        path = self._find_path(location)
        if path is None or not len(path):
            logger.warning('No path found. Return destination.')
            return [location]

        logger.info('Path: %s' % '[' + ', ' .join([location2node(grid) for grid in path]) + ']')
        path = self._find_route_node(path, step=step)
        logger.info('Path: %s' % '[' + ', ' .join([location2node(grid) for grid in path]) + ']')

        return path

    def grid_covered(self, grid, location=None):
        """
        Args:
            grid (GridInfo)
            location (list[tuple[int]]): Relative coordinate of the covered grid.

        Returns:
            list[GridInfo]:
        """
        if location is None:
            covered = [tuple(np.array(grid.location) + upper) for upper in grid.covered_grid()]
        else:
            covered = [tuple(np.array(grid.location) + upper) for upper in location]
        covered = [self[upper] for upper in covered if upper in self]
        return covered

    def missing_get(self, battle_count, mystery_count=0, siren_count=0, carrier_count=0):
        try:
            missing = self.spawn_data[battle_count].copy()
        except IndexError:
            missing = self.spawn_data[-1].copy()
        may = {'enemy': 0, 'mystery': 0, 'siren': 0, 'boss': 0, 'carrier': 0}
        missing['enemy'] -= battle_count - siren_count
        missing['mystery'] -= mystery_count
        missing['siren'] -= siren_count
        missing['carrier'] = carrier_count - self.select(is_enemy=True, may_enemy=False).count
        for grid in self:
            for attr in ['enemy', 'mystery', 'siren', 'boss']:
                if grid.__getattribute__('is_' + attr) and grid.__getattribute__('may_' + attr):
                    missing[attr] -= 1

        for grid in self:
            for upper in self.grid_covered(grid):
                for attr in ['enemy', 'mystery', 'siren', 'boss']:
                    if upper.__getattribute__('may_' + attr) and not upper.__getattribute__('is_' + attr):
                        may[attr] += 1
                if upper.may_carrier:
                    may['carrier'] += 1

        logger.attr('enemy_missing',
                    ', '.join([f'{k[:2].upper()}:{str(v).rjust(2)}' for k, v in missing.items() if k != 'battle']))
        logger.attr('enemy_may____',
                    ', '.join([f'{k[:2].upper()}:{str(v).rjust(2)}' for k, v in may.items()]))
        return may, missing

    def missing_is_none(self, battle_count, mystery_count=0, siren_count=0, carrier_count=0):
        if self.poor_map_data:
            return False

        may, missing = self.missing_get(battle_count, mystery_count, siren_count, carrier_count)

        for key in may.keys():
            if missing[key] != 0:
                return False

        return True

    def missing_predict(self, battle_count, mystery_count=0, siren_count=0, carrier_count=0):
        if self.poor_map_data:
            return False

        may, missing = self.missing_get(battle_count, mystery_count, siren_count, carrier_count)

        # predict
        for grid in self:
            for upper in self.grid_covered(grid):
                for attr in ['enemy', 'mystery', 'siren', 'boss']:
                    if upper.__getattribute__('may_' + attr) and missing[attr] > 0 and missing[attr] == may[attr]:
                        logger.info('Predict %s to be %s' % (location2node(upper.location), attr))
                        upper.__setattr__('is_' + attr, True)
                if carrier_count:
                    if upper.may_carrier and missing['carrier'] > 0 and missing['carrier'] == may['carrier']:
                        logger.info('Predict %s to be enemy' % location2node(upper.location))
                        upper.__setattr__('is_enemy', True)

    def select(self, **kwargs):
        """
        Args:
            **kwargs: Attributes of Grid.

        Returns:
            SelectedGrids:
        """
        result = []
        for grid in self:
            flag = True
            for k, v in kwargs.items():
                if grid.__getattribute__(k) != v:
                    flag = False
            if flag:
                result.append(grid)

        return SelectedGrids(result)

    def flatten(self):
        """

        Returns:
            list[GridInfo]:
        """
        return self.grids.values()
