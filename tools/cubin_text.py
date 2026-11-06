"""Fingerprint one CUDA ELF64 .text section, excluding names and debug metadata."""
import hashlib,struct


def fingerprint(blob,function):
    if len(blob)<64 or blob[:6]!=b'\x7fELF\x02\x01':raise ValueError('expected little-endian ELF64 cubin')
    offset=struct.unpack_from('<Q',blob,40)[0]
    size,count,strings=struct.unpack_from('<HHH',blob,58)
    if size!=64 or not count or strings>=count or offset+size*count>len(blob):raise ValueError('invalid section table')
    headers=[struct.unpack_from('<IIQQQQIIQQ',blob,offset+i*size) for i in range(count)]
    def section(h):
        start,length=h[4],h[5]
        if start+length>len(blob):raise ValueError('truncated section')
        return blob[start:start+length]
    names=section(headers[strings])
    for h in headers:
        if h[0]>=len(names):raise ValueError('invalid section name')
        end=names.find(b'\x00',h[0])
        if end<0:raise ValueError('unterminated section name')
        if names[h[0]:end].decode()=='.text.'+function:
            data=section(h)
            return {'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
    raise ValueError('function text section not found')
