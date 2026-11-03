#include <nmmintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
int main(int argc,char**argv){
 if(argc!=2)return 2;
 FILE*f=fopen(argv[1],"rb");if(!f)return 3;
 static unsigned char buf[8*1024*1024];uint64_t crc=0xffffffffU;size_t n;
 while((n=fread(buf,1,sizeof(buf),f))){
  size_t i=0;for(;i+8<=n;i+=8){uint64_t v;memcpy(&v,buf+i,8);crc=_mm_crc32_u64(crc,v);}
  for(;i<n;i++)crc=_mm_crc32_u8(crc,buf[i]);
 }
 if(ferror(f))return 4;fclose(f);
 printf("%08x\n",(uint32_t)crc^0xffffffffU);return 0;
}
