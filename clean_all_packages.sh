
for dir in ls -d -- *; do
    if test -d "$dir"; then
        echo $dir
	cd $dir
	make clean
	rm -r build && rm -r bin 
	cd ..
    fi
done

# make all packages
#rosmake *

#cd jaco_driver & make -j4 & cd ..

# make all packages
#rosmake *

