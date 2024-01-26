class Env:
    def __init__(self):
        self._data = {}
        self._locked = {}
        self._fallback_lock = None
        self._autolock = True

    def __setitem__(self, key, item):
        self.set(key, item)

    def __getitem__(self, key):
        return self.get(key)

    def __repr__(self):
        return repr(self._data)

    def __len__(self):
        return len(self._data)

    def __delitem__(self, key):
        if self.is_locked(key):
            raise ValueError(f"Data with key '{key}' is locked!")
        del self._data[key]
        
    def __cmp__(self, dict_):
        return self.__cmp__(self._data, dict_)

    def __contains__(self, item):
        return item in self._data

    def __iter__(self):
        return iter(self._data)
        
    def lock(self, key, state):
        self._locked[key] = state
        
    def is_locked(self, key):
        return self._locked.get(key, self._fallback_lock) is True
                
    def set(self, key, value):
        if self.is_locked(key):
            raise ValueError(f"Data with key '{key}' is locked!")
        self._data[key] = value
        
    def get(self, key, fallback='_no_fallback_'):
        if self._autolock and self._fallback_lock is None:
            self._fallback_lock = True  # As soon as any field is read - lock data from changes
        if key not in self._data:
            if fallback == '_no_fallback_':
                    raise ValueError(f"No key '{key}' in data!")
            else:
                return fallback
        else:
            return self._data[key]
        
    def keys(self):
        return self._data.keys()
    
    def items(self):
        return self._data.items()

    def set_dict(self, data):
        for k,v in data.items():
            self.set(k, v)
            
    def lock_dict(self, data):
        for k, v in data.items():
            self.lock(k, not v is False)
            
    def as_dict(self):
        return {**self._data}


ENV = Env()


ENV__DEBUG_PRINT = 'ENV__DEBUG_PRINT'
ENV[ENV__DEBUG_PRINT] = True
