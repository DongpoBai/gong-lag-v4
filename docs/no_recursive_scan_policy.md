# No-recursive-scan policy for Rhodes/CARC project trees

Do not run recursive filesystem scans in Rhodes project directories.

Forbidden on `/project2/erhodes_44`, Rhodes project roots, or other multi-TB project trees:

- `find /project2/erhodes_44 ...`
- `grep -R /project2/erhodes_44 ...`
- `du -sh /project2/erhodes_44/...` on broad directories
- any recursive search over unknown large trees

Allowed:

- exact-path `ls -ld`
- exact-path `ls -lh`
- exact-path `stat`
- exact-file `head`, `sed -n`, `wc -l`
- `git grep` inside the small Git repository
- inspection of one user-provided candidate file
- reading curated manifests and explicit paths

Reason:

The Rhodes/CARC project trees may contain tens of TB of data. Recursive scans are unsafe,
slow, and unnecessary for this project.
