import struct
import json

with open('box_pointcloud.glb', 'rb') as f:
    magic, version, total_length = struct.unpack('<III', f.read(12))
    print(f'Magic: {hex(magic)} (expected 0x46546c67)')
    print(f'Version: {version}')
    print(f'Total length: {total_length}')
    
    json_chunk_len, json_chunk_type = struct.unpack('<II', f.read(8))
    print(f'JSON chunk length: {json_chunk_len}')
    print(f'JSON chunk type: {hex(json_chunk_type)} (expected 0x4e4f534a)')
    
    json_data = f.read(json_chunk_len).decode('utf-8').rstrip('\x00')
    print(f'JSON data length: {len(json_data)}')
    print('JSON content:')
    print(json_data[:800])
    print('...')
    
    bin_chunk_len, bin_chunk_type = struct.unpack('<II', f.read(8))
    print(f'\nBIN chunk length: {bin_chunk_len}')
    print(f'BIN chunk type: {hex(bin_chunk_type)} (expected 0x004e4942)')
    
    json_obj = json.loads(json_data)
    print(f'\nParsed JSON:')
    print(f'  accessors: {len(json_obj.get("accessors", []))}')
    print(f'  bufferViews: {len(json_obj.get("bufferViews", []))}')
    
    if 'accessors' in json_obj:
        for i, acc in enumerate(json_obj['accessors']):
            print(f'  accessor[{i}]: count={acc.get("count")}, type={acc.get("type")}, bufferView={acc.get("bufferView")}')
    
    if 'bufferViews' in json_obj:
        for i, bv in enumerate(json_obj['bufferViews']):
            print(f'  bufferView[{i}]: byteOffset={bv.get("byteOffset")}, byteLength={bv.get("byteLength")}')
    
    bin_data_start = f.tell()
    print(f'\nBIN data starts at file offset: {bin_data_start}')
    
    if 'bufferViews' in json_obj and 'accessors' in json_obj:
        bv = json_obj['bufferViews'][0]
        acc = json_obj['accessors'][0]
        byte_offset = bv.get('byteOffset', 0)
        count = acc.get('count', 0)
        print(f'\nVertex data:')
        print(f'  byteOffset in buffer: {byte_offset}')
        print(f'  vertex count: {count}')
        
        f.seek(bin_data_start + byte_offset)
        total_vertices = count
        particle_count = total_vertices // 2
        
        print(f'  total vertices: {total_vertices}')
        print(f'  particle count: {particle_count}')
        
        vertices = []
        for i in range(min(10, total_vertices)):
            x, y, z = struct.unpack('<fff', f.read(12))
            vertices.append((x, y, z))
        
        print(f'\nFirst 5 position vertices:')
        for i, v in enumerate(vertices[:5]):
            print(f'  vertex[{i}]: ({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f})')
        
        print(f'\nFirst 5 sk vertices (at offset {particle_count}):')
        f.seek(bin_data_start + byte_offset + particle_count * 12)
        for i in range(5):
            x, y, z = struct.unpack('<fff', f.read(12))
            print(f'  sk_vertex[{i}]: x={x:.6e}, y={y:.6e}, z={z:.6e}')
