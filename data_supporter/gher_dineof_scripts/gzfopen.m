% fid=gzfopen(fname,permisson,machineformat)
% 
% same as fopen except that the file is decompressed if it ends 
% with ".gz"
%
%
% Alexander Barth, 2008-03-19


function fid=gzfopen(fname,permisson,machineformat)

global GZ_FILE_IDS GZ_FILE_NAMES

if isempty(GZ_FILE_NAMES)
  GZ_FILE_NAMES={};
end

[fname] = gread_tilde_expand(fname);

zipped = 0;

if length(fname) > 3
  if strcmp(fname(end-2:end),'.gz') 
     zipped = 1;
  end
end

if zipped
  tmpdir = getenv('TMPDIR');

  if isempty(tmpdir)
    tmpdir = '/tmp';
  end
  
  tmp = tempname(tmpdir);
  %system(['cp -i ' fname '  ' tmp '.gz;   gunzip ' tmp '.gz; ']);
  
  cmd = ['gunzip --to-stdout "' fname '"  >  "' tmp '"'];
  
  system(cmd);
  fid = fopen(tmp,permisson,machineformat);

  GZ_FILE_IDS(end+1) = fid;
  GZ_FILE_NAMES{end+1} = tmp;
else
  fid = fopen(fname,permisson,machineformat);
end

