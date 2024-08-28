# Retargeting with Graph Positional Encodings

## Local Development
### Building the Singularity image on Mac
First install the Vagrant VM
```shell
brew install virtualbox && \
    brew install vagrant && \
    brew install vagrant-manager
```

Next start up the vm in the root directory of the project `retarget/` and run
```shell
export VM=sylabs/singularity-3.0-ubuntu-bionic64 && \
    vagrant init $VM && \
    vagrant up && \
    vagrant ssh
```

Once you are in the VM navigate to `/vagrant` and run
```
singularity build --fakeroot container.sif singularity.def
```
