#! /usr/bin/bash

tag=1.13.0

zip_name="v$tag.zip"
src_dir="googletest-$tag"

if [ $1 == "install" ]
then  
  if [ ! -f gtest_main.a -o ! -f gtest.a ]
  then
    echo "Installing Googletest"

    if [ ! -f $zip_name ]
    then
      wget https://github.com/google/googletest/archive/refs/tags/$zip_name
    fi

    if [ ! -d $src_dir ]
    then
      unzip $zip_name
    fi  

    make

    if [ -h gtest ]
    then
      rm gtest
    fi

    ln -s $src_dir/googletest/include/gtest gtest
      
    echo Installation of Googletest is complete
  else
    echo "Googletest is already installed"  
  fi

elif [ $1 == "clean" ]
then 
  rm -f $zip_name gtest gtest.a gtest-all.o gtest_main.a gtest_main.o
  rm -rf $src_dir
  echo "All installation files for Googletest have been deleted"  
fi