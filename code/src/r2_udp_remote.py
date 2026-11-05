"""Bounded UDP endpoint for switch experiments. No privileged operations."""
import json
import select
import socket
import struct
import sys
import time

def main():
    spec=json.loads(sys.stdin.readline())
    mode=spec['mode']
    duration=float(spec['duration_s'])
    if not 0<duration<=60:
        raise ValueError('Bounded experiment duration required')
    sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,4*1024*1024)
    sock.bind((spec['bind_ip'],int(spec.get('bind_port',0))))
    print(json.dumps({'ready':True,'address':sock.getsockname()}),flush=True)
    if mode=='server':
        allowed=set(spec['allowed_sources'])
        sock.settimeout(.1)
        start=time.monotonic()
        packets=octets=probes=0
        while time.monotonic()-start<duration:
            try:
                data,peer=sock.recvfrom(65536)
            except socket.timeout:
                continue
            if peer[0] not in allowed:
                continue
            packets+=1
            octets+=len(data)
            if data[:1]==b'P':
                sock.sendto(data,peer)
                probes+=1
        result=dict(packets=packets,payload_bytes=octets,echoed=probes,elapsed_s=time.monotonic()-start)
    elif mode=='load':
        target=(spec['target_ip'],int(spec['target_port']))
        size=int(spec['payload_bytes'])
        if size!=1400 or not 0<=spec['wire_mbps']<=110:
            raise ValueError('Unsupported bounded load')
        payload=b'L'+bytes(size-1)
        rate=spec['wire_mbps']*1e6/(8*(size+66))
        start=time.monotonic()
        sent=0
        while time.monotonic()-start<duration:
            elapsed=time.monotonic()-start
            due=int(elapsed*rate)
            # Catch up short scheduling delays while bounding any one burst.
            for _ in range(min(256,max(0,due-sent))):
                sock.sendto(payload,target)
                sent+=1
            time.sleep(.0005)
        result=dict(sent=sent,payload_bytes=sent*size,wire_bytes=sent*(size+66),elapsed_s=time.monotonic()-start,
                    requested_wire_mbps=spec['wire_mbps'])
    elif mode=='probe':
        sock.setblocking(False)
        target=(spec['target_ip'],int(spec['target_port']))
        rate=100.
        size=256
        start=time.monotonic()
        sent={}
        returned={}
        next_send=start
        while time.monotonic()<start+duration+2:
            now=time.monotonic()
            if now<start+duration and now>=next_send:
                seq=len(sent)
                packet=b'P'+struct.pack('!I',seq)+bytes(size-5)
                sent[seq]=time.monotonic()
                sock.sendto(packet,target)
                next_send=start+len(sent)/rate
            available,_,_=select.select([sock],[],[],.001)
            if available:
                data,peer=sock.recvfrom(65536)
                arrival=time.monotonic()
                if peer==target and data[:1]==b'P' and len(data)>=5:
                    seq=struct.unpack('!I',data[1:5])[0]
                    if seq in sent and seq not in returned:
                        returned[seq]=arrival
        result=dict(probes=[dict(sequence=seq,send_s=stamp-start,rtt_s=returned[seq]-stamp if seq in returned else None)
                            for seq,stamp in sent.items()],sent=len(sent),received=len(returned),duration_s=duration)
    else:
        raise ValueError('Unknown mode')
    sock.close()
    print(json.dumps(result,separators=(',',':')),flush=True)

if __name__=='__main__':
    main()
