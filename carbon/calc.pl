#!/usr/bin/env perl

use strict;
use warnings;

##### config

# carbon in local grid: kgCO₂eq/kWh (https://app.electricitymaps.com)
my $cfg_effic = 0.2120;

# PUE coefficient (Strubell et al., 2020)
my $cfg_pue_coeff = 1.58;

# Hardware consumption in watts
my $cfg_power = 250;

##### end-of-config

my $agg_time = 0;
my $agg_carbon = 0;
my $count = 0;
my %data = ();
my @sorted_keys = ();

while(<STDIN>)
{
    chomp;
    $count++;

    my $power = $cfg_power;
    my $effic = $cfg_effic;
    my $pue_coeff = $cfg_pue_coeff;

    my @fields = split(/\t/, $_);
    my @time = split(/m/, $fields[1]);

    if($fields[0] eq "crf")
    {
        $pue_coeff = 1;
        $power = 19;
    }

    # training time
    my $hours = (($time[0] * 60) + $time[1]) / 3600;

    # power consumption in kWh = P(W) × t(h) / 1000
    my $power_consum = ($hours * $power) / 1000;

    # carbon emissions
    my $carbon = $effic * $power_consum * $pue_coeff;

    $agg_time += $hours;
    $agg_carbon += $carbon;

    $data{$fields[0]} = [$hours, $carbon];
    push(@sorted_keys, $fields[0]);
}

my $carbon_mean = $agg_carbon/$count;
my $time_mean = $agg_time/$count;
foreach my $model (@sorted_keys)
{
    my $dev = (($data{$model}[1] - $carbon_mean)/$carbon_mean)*100;
    printf("%s\t%.2f\t%.6f\t%.8f\n", $model, $data{$model}[0], $data{$model}[1], $dev);
}


printf("Total\t%.2f\t%.6f\n\n", $agg_time, $agg_carbon);
printf("Average emissions: %.6f\n", $carbon_mean);
printf("Average training time: %.2f\n", $time_mean);
