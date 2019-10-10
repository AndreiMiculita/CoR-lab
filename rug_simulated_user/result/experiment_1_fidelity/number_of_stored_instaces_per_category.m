figure ();
hold on;
grid on;
NIC= [21,4,6,26,20,7,4,10,17,7,10,7];
list = {'sponge', 'shampoo', 'can-food', 'camera', 'greens', 'lightbulb', 'mushroom', 'lime', 'comb', 'bell-pepper', 'toothpaste', 'orange'};
categories = categorical(list,list);
bar (categories, NIC);
ylabel('Number of Stored Instances','FontSize',15);