all: start cifar10 cifar100

start:
	@bash start.sh

cifar10:
	@nohup bash commands/cifar10.sh 0 30 &

cifar100:
	@nohup bash commands/cifar100.sh 0 30 &
